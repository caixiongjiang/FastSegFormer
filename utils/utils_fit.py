import os

import torch
from nets.unet_training import CE_Loss, Dice_loss, Focal_Loss
from tqdm import tqdm

from utils.utils import get_lr
from utils.utils_metrics import f_score

import torch.nn.functional as F

from utils.distillation_loss import NFD_loss_after_conv1x1
from utils.distillation_loss import kl_pixel_loss


def fit_one_epoch(model_train, model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen, gen_val, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0

    val_loss        = 0
    val_f_score     = 0


    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:
            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():

                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes=num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)

    model_train.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)


            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss  = loss + main_dice
            _f_score    = f_score(outputs, labels)

            val_loss    += loss.item()
            val_f_score += _f_score.item()
            
        if local_rank == 0:
            pbar.set_postfix(**{'val_loss'  : val_loss / (iteration + 1),
                                'f_score'   : val_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)
            
    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step, val_loss/ epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, model_train)
        print('Epoch:'+ str(epoch+1) + '/' + str(Epoch))
        print('Total Loss: %.3f || Val Loss: %.3f ' % (total_loss / epoch_step, val_loss / epoch_step_val))
        
        #-----------------------------------------------#
        #   保存权值
        #-----------------------------------------------#
        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f-val_loss%.3f.pth'%((epoch + 1), total_loss / epoch_step, val_loss / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))


def fit_one_epoch_distillation(Multi_resolution, s_model_train, s_model, t_model_eval, t_model, loss_history, eval_callback, optimizer, epoch, epoch_step, epoch_step_val, gen,
                  gen_val, Epoch, cuda, NFD_loss, KL_pixel_loss, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period,
                  save_dir, local_rank=0):
    total_loss = 0
    total_f_score = 0
    total_label_loss_train = 0

    val_loss = 0
    val_f_score = 0
    total_label_loss_val = 0

    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)
    s_model_train.train()
    t_model_eval.eval()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step:
            break
        t_imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                t_imgs = t_imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if Multi_resolution:
            s_imgs = F.interpolate(t_imgs, size=(224, 224), mode='bilinear', align_corners=True)

        if not fp16:
            if Multi_resolution:
                s_features_out = s_model_train(s_imgs)
            else:
                s_features_out = s_model_train(t_imgs)
            t_features_out = t_model_eval(t_imgs)


            if focal_loss:
                label_loss_train = Focal_Loss(s_features_out[-1], pngs, weights, num_classes=num_classes)
            else:
                label_loss_train = CE_Loss(s_features_out[-1], pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(s_features_out[-1], labels)
                label_loss_train = label_loss_train + main_dice

            # print(label_loss_train)

            # distillation loss
            loss = label_loss_train

            if NFD_loss:
                normal_loss = 0
                for i in range(len(s_features_out)):
                    if i == 7:
                        break
                    _, t_C, _, _ = t_features_out[i].shape
                    _, s_C, _, _ = s_features_out[i].shape
                    normal_loss_optimizer = NFD_loss_after_conv1x1(t_C, s_C)
                    normal_loss_optimizer.cuda()
                    normal_loss = normal_loss_optimizer(t_features_out[i], s_features_out[i])

                # print(normal_loss)
                loss = loss + 5 * normal_loss

            if KL_pixel_loss:
                kl_loss = kl_pixel_loss(t_features_out[-1], s_features_out[-1])
                # print(kl_loss)
                loss = loss + 0.5 * kl_loss


            with torch.no_grad():
                _f_score = f_score(s_features_out[-1], labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():
                s_features_out = s_model_train(s_imgs)
                t_features_out = t_model_eval(t_imgs)


                if focal_loss:
                    label_loss_train = Focal_Loss(s_features_out[-1], pngs, weights, num_classes=num_classes)
                else:
                    label_loss_train = CE_Loss(s_features_out[-1], pngs, weights, num_classes=num_classes)

                if dice_loss:
                    main_dice = Dice_loss(s_features_out[-1], labels)
                    label_loss_train = label_loss_train + main_dice

                # print(label_loss_train)

                # distillation loss
                loss = label_loss_train

                if NFD_loss:
                    normal_loss = 0
                    for i in range(len(s_features_out)):
                        if i == 7:
                            break
                        _, t_C, _, _ = t_features_out[i].shape
                        _, s_C, _, _ = s_features_out[i].shape
                        normal_loss_optimizer = NFD_loss_after_conv1x1(t_C, s_C)
                        normal_loss_optimizer.cuda()
                        normal_loss = normal_loss_optimizer(t_features_out[i], s_features_out[i])

                    # print(normal_loss)
                    loss = loss + 5 * normal_loss

                if KL_pixel_loss:
                    kl_loss = kl_pixel_loss(t_features_out[-1], s_features_out[-1])
                    # print(kl_loss)
                    loss = loss + 0.5 * kl_loss

                with torch.no_grad():
                    # -------------------------------#
                    #   计算f_score
                    # -------------------------------#
                    _f_score = f_score(s_features_out[-1], labels)

            # ----------------------#
            #   反向传播
            # ----------------------#
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss += loss.item()
        total_label_loss_train += label_loss_train.item()
        total_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'stu_total_loss': total_loss / (iteration + 1),
                                'stu_f_score': total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Train')
        print('Start Validation')
        pbar = tqdm(total=epoch_step_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3)

    s_model_train.eval()
    t_model_eval.eval()
    for iteration, batch in enumerate(gen_val):
        if iteration >= epoch_step_val:
            break
        t_imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                t_imgs = t_imgs.cuda(local_rank)
                pngs = pngs.cuda(local_rank)
                labels = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

            if Multi_resolution:
                s_imgs = F.interpolate(t_imgs, size=(224, 224), mode='bilinear', align_corners=True)

            if Multi_resolution:
                s_features_out = s_model_train(s_imgs)
            else:
                s_features_out = s_model_train(t_imgs)
            t_features_out = t_model_eval(t_imgs)

            if focal_loss:
                label_loss_val = Focal_Loss(s_features_out[-1], pngs, weights, num_classes=num_classes)
            else:
                label_loss_val = CE_Loss(s_features_out[-1], pngs, weights, num_classes=num_classes)

            if dice_loss:
                main_dice = Dice_loss(s_features_out[-1], labels)
                label_loss_val = label_loss_val + main_dice

            # print(label_loss)

            # distillation loss
            loss = label_loss_val

            if NFD_loss:
                normal_loss = 0
                for i in range(len(s_features_out)):
                    if i == 7:
                        break
                    _, t_C, _, _ = t_features_out[i].shape
                    _, s_C, _, _ = s_features_out[i].shape
                    normal_loss_optimizer = NFD_loss_after_conv1x1(t_C, s_C)
                    normal_loss_optimizer.cuda()
                    normal_loss = normal_loss_optimizer(t_features_out[i], s_features_out[i])

                # print(normal_loss)
                loss = loss + 5 * normal_loss


            if KL_pixel_loss:
                kl_loss = kl_pixel_loss(t_features_out[-1], s_features_out[-1])
                # print(kl_loss)
                loss = loss + 0.5 * kl_loss

            _f_score = f_score(s_features_out[-1], labels)

            val_loss += loss.item()
            total_label_loss_val += label_loss_val.item()
            val_f_score += _f_score.item()

        if local_rank == 0:
            pbar.set_postfix(**{'stu_val_loss': val_loss / (iteration + 1),
                                'stu_f_score': val_f_score / (iteration + 1),
                                'stu_lr': get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        print('Finish Validation')
        loss_history.append_loss(epoch + 1, total_loss / epoch_step, val_loss / epoch_step_val)
        eval_callback.on_epoch_end(epoch + 1, s_model_train)
        print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f, Label loss_train:%.3f, || Val Loss: %.3f, Label loss_val:%.3f, ' % (total_loss / epoch_step, total_label_loss_train / epoch_step, val_loss / epoch_step_val, total_label_loss_val / epoch_step_val))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(s_model.state_dict(), os.path.join(save_dir, 'ep%03d-label_loss%.3f-label_loss_val%.3f.pth' % (
            (epoch + 1), total_label_loss_train / epoch_step, total_label_loss_val / epoch_step_val)))

        if len(loss_history.val_loss) <= 1 or (val_loss / epoch_step_val) <= min(loss_history.val_loss):
            print('Save best model to best_epoch_weights.pth')
            torch.save(s_model.state_dict(), os.path.join(save_dir, "Stu_best_epoch_weights.pth"))

        torch.save(s_model.state_dict(), os.path.join(save_dir, "Stu_last_epoch_weights.pth"))





def fit_one_epoch_no_val(model_train, model, loss_history, optimizer, epoch, epoch_step, gen, Epoch, cuda, dice_loss, focal_loss, cls_weights, num_classes, fp16, scaler, save_period, save_dir, local_rank=0):
    total_loss      = 0
    total_f_score   = 0
    
    if local_rank == 0:
        print('Start Train')
        pbar = tqdm(total=epoch_step,desc=f'Epoch {epoch + 1}/{Epoch}',postfix=dict,mininterval=0.3)
    model_train.train()
    for iteration, batch in enumerate(gen):
        if iteration >= epoch_step: 
            break
        imgs, pngs, labels = batch
        with torch.no_grad():
            weights = torch.from_numpy(cls_weights)
            if cuda:
                imgs    = imgs.cuda(local_rank)
                pngs    = pngs.cuda(local_rank)
                labels  = labels.cuda(local_rank)
                weights = weights.cuda(local_rank)

        optimizer.zero_grad()
        if not fp16:

            outputs = model_train(imgs)

            if focal_loss:
                loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
            else:
                loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

            if dice_loss:
                main_dice = Dice_loss(outputs, labels)
                loss      = loss + main_dice

            with torch.no_grad():

                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()
        else:
            from torch.cuda.amp import autocast
            with autocast():

                outputs = model_train(imgs)

                if focal_loss:
                    loss = Focal_Loss(outputs, pngs, weights, num_classes = num_classes)
                else:
                    loss = CE_Loss(outputs, pngs, weights, num_classes = num_classes)

                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss      = loss + main_dice

                with torch.no_grad():
                    _f_score = f_score(outputs, labels)

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

        total_loss      += loss.item()
        total_f_score   += _f_score.item()
        
        if local_rank == 0:
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1), 
                                'f_score'   : total_f_score / (iteration + 1),
                                'lr'        : get_lr(optimizer)})
            pbar.update(1)

    if local_rank == 0:
        pbar.close()
        loss_history.append_loss(epoch + 1, total_loss/ epoch_step)
        print('Epoch:'+ str(epoch + 1) + '/' + str(Epoch))
        print('Total Loss: %.3f' % (total_loss / epoch_step))

        if (epoch + 1) % save_period == 0 or epoch + 1 == Epoch:
            torch.save(model.state_dict(), os.path.join(save_dir, 'ep%03d-loss%.3f.pth'%((epoch + 1), total_loss / epoch_step)))

        if len(loss_history.losses) <= 1 or (total_loss / epoch_step) <= min(loss_history.losses):
            print('Save best model to best_epoch_weights.pth')
            torch.save(model.state_dict(), os.path.join(save_dir, "best_epoch_weights.pth"))
            
        torch.save(model.state_dict(), os.path.join(save_dir, "last_epoch_weights.pth"))





