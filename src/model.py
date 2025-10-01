from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_mnist_model(
    max_epochs: int = 1, random_state: int = 42
) -> Pipeline:
    """Создает MLP модель для MNIST классификации."""
    if max_epochs < 1:
        max_epochs = 1

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    # Два скрытых слоя: 128 и 64 нейрона
                    hidden_layer_sizes=(128, 64),
                    activation='relu',              # ReLU активация
                    solver='adam',                  # Adam оптимизатор
                    alpha=1e-4,                     # L2 регуляризация
                    # Начальная скорость обучения
                    learning_rate_init=0.001,
                    # Максимальное количество эпох
                    max_iter=max_epochs,
                    early_stopping=True,           # Ранняя остановка
                    # 10% данных для валидации
                    validation_fraction=0.1,
                    # Остановка через 10 эпох без улучшения
                    n_iter_no_change=10,
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipeline
