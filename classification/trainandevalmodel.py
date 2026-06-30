from classification.model_copy import ModelMLP, ModelSVM
from classification.eval import plot_confusion_matrix, plot_training_loss, evaluate_classifier


def train_model_for_part(split_data, subject, label_map, do_8_channels = False, path_to_output = None, use_freq=True):
    """
    Launch model training for one participant using training data
    """
    my_model = ModelMLP()
    model_name = f"MLP_8_{do_8_channels}_use_freq_{use_freq}"
    my_model.fit(split_data[subject]['training'], split_data[subject]['testing'])
    if path_to_output is not None:
        my_model.save(f"{path_to_output}/{model_name}_{subject}.pth")

    # ensure class names are ordered by label index
    if isinstance(label_map, dict):
        class_names = [k for k, v in sorted(label_map.items(), key=lambda kv: kv[1])]
    else:
        class_names = label_map

    metrics, output = evaluate_classifier(my_model, split_data[subject]['testing']['x'], split_data[subject]['testing']['y'], class_names)

    if path_to_output is not None:
        with open(f"{path_to_output}/{model_name}_report_{subject}.txt", "w", encoding="utf-8") as f:
            f.write("\n".join(output))

    fig = plot_confusion_matrix(metrics["y_true"], metrics["y_pred"], class_names=class_names)
    if fig is not None and path_to_output is not None:
        fig.savefig(f"{path_to_output}/{model_name}_Confusion_Matrix_{subject}.png")

    loss_fig = plot_training_loss(my_model)
    if loss_fig is not None and path_to_output is not None:
        loss_fig.savefig(f"{path_to_output}/{model_name}_Training_loss_{subject}.png")

        