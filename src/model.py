from typing import Optional

from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def create_mnist_model(max_epochs: int = 1, random_state: int = 42) -> Pipeline:
    """Создает MLP модель для MNIST классификации."""
    if max_epochs < 1:
        max_epochs = 1

    pipeline = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(128, 64),  # Два скрытых слоя: 128 и 64 нейрона
                    activation='relu',              # ReLU активация
                    solver='adam',                  # Adam оптимизатор
                    alpha=1e-4,                     # L2 регуляризация
                    learning_rate_init=0.001,       # Начальная скорость обучения
                    max_iter=max_epochs,            # Максимальное количество эпох
                    early_stopping=True,           # Ранняя остановка
                    validation_fraction=0.1,       # 10% данных для валидации
                    n_iter_no_change=10,            # Остановка через 10 эпох без улучшения
                    random_state=random_state,
                ),
            ),
        ]
    )
    return pipeline 