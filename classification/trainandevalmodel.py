from classification.model_copy import ModelMLP
from classification.eval import plot_confusion_matrix, plot_training_loss, evaluate_classifier


def train_model_for_part(split_data, subject, label_map, path_to_output = None, ):
    """
    Launch model training for one participant using training data
    """
    my_model = ModelMLP()
    my_model.fit(split_data[subject]['training'], split_data[subject]['testing'])
    if path_to_output is not None:
        my_model.save(f"{path_to_output}/MLP_{subject}.pth")

    # ensure class names are ordered by label index
    if isinstance(label_map, dict):
        class_names = [k for k, v in sorted(label_map.items(), key=lambda kv: kv[1])]
    else:
        class_names = label_map

    metrics = evaluate_classifier(my_model, split_data[subject]['testing']['x'], split_data[subject]['testing']['y'], class_names)

    fig = plot_confusion_matrix(metrics["y_true"], metrics["y_pred"], class_names=class_names)
    if fig is not None and path_to_output is not None:
        fig.savefig(f"{path_to_output}/Confusion_Matrix_{subject}.png")

    loss_fig = plot_training_loss(my_model)
    if loss_fig is not None and path_to_output is not None:
        loss_fig.savefig(f"{path_to_output}/Training_loss_{subject}.png")

        