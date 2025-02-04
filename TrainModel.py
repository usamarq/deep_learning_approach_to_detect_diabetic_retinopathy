
import os
import sys
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import cohen_kappa_score, precision_score, recall_score, accuracy_score
from Visualizations import plot_confusion_matrix
from tqdm import tqdm


def train_model(model, train_loader, val_loader, device, criterion, optimizer, lr_scheduler, num_epochs=25,
                checkpoint_path='model.pth'):
    best_model = model.state_dict()
    best_epoch = None
    best_val_kappa = -1.0  # Initialize the best kappa score
    train_losses = []
    train_accuracies = []  
    val_kappas = []
    val_losses = [] 
    val_accuracies = []

    for epoch in range(1, num_epochs + 1):
        print(f'\nEpoch {epoch}/{num_epochs}')
        running_loss = []
        all_preds = []
        all_labels = []

        model.train()

        with tqdm(total=len(train_loader), desc=f'Training', unit=' batch', file=sys.stdout) as pbar:
            for images, labels in train_loader:
                if not isinstance(images, list):
                    images = images.to(device)  # single image case
                else:
                    images = [x.to(device) for x in images]  # dual images case

                labels = labels.to(device)

                optimizer.zero_grad()

                outputs = model(images)
                loss = criterion(outputs, labels.long())

                loss.backward()
                optimizer.step()

                preds = torch.argmax(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                running_loss.append(loss.item())

                pbar.set_postfix({'lr': f'{optimizer.param_groups[0]["lr"]:.1e}', 'Loss': f'{loss.item():.4f}'})
                pbar.update(1)

        lr_scheduler.step()

        epoch_loss = sum(running_loss) / len(running_loss)
        train_losses.append(epoch_loss)

        epoch_accuracy = accuracy_score(all_labels, all_preds)
        train_accuracies.append(epoch_accuracy)

        train_metrics = compute_metrics(all_preds, all_labels, per_class=True)
        kappa, accuracy, precision, recall = train_metrics[:4]

        print(f'[Train] Kappa: {kappa:.4f} Accuracy: {accuracy:.4f} '
              f'Precision: {precision:.4f} Recall: {recall:.4f} Loss: {epoch_loss:.4f}')

        if len(train_metrics) > 4:
            precision_per_class, recall_per_class = train_metrics[4:]
            for i, (precision, recall) in enumerate(zip(precision_per_class, recall_per_class)):
                print(f'[Train] Class {i}: Precision: {precision:.4f}, Recall: {recall:.4f}')

        
        # Evaluation on the validation set at the end of each epoch
        val_loss, val_metrics = evaluate_model(model, val_loader, device, return_loss=True)
        val_kappa, val_accuracy, val_precision, val_recall = val_metrics[:4]
        val_losses.append(val_loss)
        val_kappas.append(val_kappa)
        val_accuracies.append(val_accuracy)
        print(f'[Val] Kappa: {val_kappa:.4f} Accuracy: {val_accuracy:.4f} '
              f'Precision: {val_precision:.4f} Recall: {val_recall:.4f}')

        if val_kappa > best_val_kappa:
            best_val_kappa = val_kappa
            best_epoch = epoch
            best_model = model.state_dict()
            torch.save(best_model, checkpoint_path)

    print(f'[Val] Best kappa: {best_val_kappa:.4f}, Epoch {best_epoch}')
    print("\nFinal Confusion Matrix on Validation Set:")

    evaluate_model(model, val_loader, device, show_confusion_matrix=True, return_loss=True)
    
    return model, train_losses, val_kappas, val_losses, train_accuracies, val_accuracies

def evaluate_model(model, test_loader, device, test_only=False, show_confusion_matrix=False, return_loss=False, prediction_path='./test_predictions_ResNet18.csv'):
    model.eval()

    all_preds = []
    all_labels = []
    all_image_ids = []
    total_loss = 0 
    criterion = nn.CrossEntropyLoss()  

    with tqdm(total=len(test_loader), desc=f'Evaluating', unit=' batch', file=sys.stdout) as pbar:
        for i, data in enumerate(test_loader):

            if test_only:
                images = data
                labels = None
            else:
                images, labels = data

            if not isinstance(images, list):
                images = images.to(device)  # single image case
            else:
                images = [x.to(device) for x in images]  # dual images case

            if labels is not None: 
                labels = labels.to(device)

            with torch.no_grad():
                outputs = model(images)
                preds = torch.argmax(outputs, 1)

                # Compute loss for validation if requested
                if return_loss and labels is not None:
                    loss = criterion(outputs, labels.long())
                    total_loss += loss.item()

            if not isinstance(images, list):
                # single image case
                all_preds.extend(preds.cpu().numpy())
                image_ids = [
                    os.path.basename(test_loader.dataset.data[idx]['img_path']) for idx in
                    range(i * test_loader.batch_size, i * test_loader.batch_size + len(images))
                ]
                all_image_ids.extend(image_ids)
                if not test_only:
                    all_labels.extend(labels.cpu().numpy())
            else:
                # dual images case
                for k in range(2):
                    all_preds.extend(preds.cpu().numpy())
                    image_ids = [
                        os.path.basename(test_loader.dataset.data[idx][f'img_path{k + 1}']) for idx in
                        range(i * test_loader.batch_size, i * test_loader.batch_size + len(images[k]))
                    ]
                    all_image_ids.extend(image_ids)
                    if not test_only:
                        all_labels.extend(labels.cpu().numpy())

            pbar.update(1)

    # Save predictions to csv file for Kaggle online evaluation
    if test_only:
        df = pd.DataFrame({
            'ID': all_image_ids,
            'TARGET': all_preds
        })
        df.to_csv(prediction_path, index=False)
        print(f'[Test] Save predictions to {os.path.abspath(prediction_path)}')
    elif show_confusion_matrix and not test_only:
        plot_confusion_matrix(all_labels, all_preds, class_names=['No DR', 'Mild', 'Moderate', 'Severe', 'PDR'])
    else:
        metrics = compute_metrics(all_preds, all_labels)
        if return_loss:
            avg_loss = total_loss / len(test_loader)
            return avg_loss, metrics
        return metrics

def compute_metrics(preds, labels, per_class=False):
    kappa = cohen_kappa_score(labels, preds, weights='quadratic')
    accuracy = accuracy_score(labels, preds)
    precision = precision_score(labels, preds, average='weighted', zero_division=0)
    recall = recall_score(labels, preds, average='weighted', zero_division=0)

    # Calculate and print precision and recall for each class
    if per_class:
        precision_per_class = precision_score(labels, preds, average=None, zero_division=0)
        recall_per_class = recall_score(labels, preds, average=None, zero_division=0)
        return kappa, accuracy, precision, recall, precision_per_class, recall_per_class

    return kappa, accuracy, precision, recall
