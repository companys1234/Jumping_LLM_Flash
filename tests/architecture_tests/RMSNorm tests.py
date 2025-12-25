import pytest
import torch
import math
from torch import nn
from RMSNorm import RMSNorm


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.ones(dim))

    def forward(self, x):
        norm = x.norm(dim=-1, keepdim=True) / math.sqrt(x.shape[-1])
        return self.scale * x / (norm + self.eps)



class TestRMSNorm:

    @pytest.fixture
    def rms_norm(self):
        """Фикстура для создания экземпляра RMSNorm"""
        return RMSNorm(dim=64)

    @pytest.fixture
    def rms_norm_with_custom_eps(self):
        """Фикстура для создания RMSNorm с пользовательским eps"""
        return RMSNorm(dim=32, eps=1e-6)

    def test_initialization(self):
        """Тест инициализации параметров"""
        # Проверка размерности параметра scale
        norm = RMSNorm(dim=128)
        assert norm.scale.shape == (128,)
        assert norm.eps == 1e-8

        # Проверка инициализации scale единицами
        assert torch.allclose(norm.scale, torch.ones(128))

    def test_initialization_custom_eps(self):
        """Тест инициализации с пользовательским eps"""
        norm = RMSNorm(dim=64, eps=1e-6)
        assert norm.eps == 1e-6

    def test_forward_shape(self, rms_norm):
        """Тест сохранения формы тензора после forward"""
        batch_size, seq_len, dim = 4, 10, 64

        # Создаем случайный тензор
        x = torch.randn(batch_size, seq_len, dim)

        # Применяем RMSNorm
        output = rms_norm(x)

        # Проверяем сохранение формы
        assert output.shape == x.shape

    def test_forward_values(self):
        """Тест правильности вычислений RMSNorm"""
        # Создаем простой тензор для проверки вычислений
        dim = 4
        norm = RMSNorm(dim=dim)

        # Отключаем обучение scale для предсказуемости
        norm.scale.data = torch.ones(dim)

        # Тестовый тензор: [batch_size=2, seq_len=1, dim=4]
        x = torch.tensor([[[1.0, 2.0, 3.0, 4.0]],
                          [[-1.0, -2.0, -3.0, -4.0]]])

        # Вычисляем RMSNorm вручную для проверки
        output = norm(x)

        # Вычисляем RMS вручную
        rms_manual_1 = math.sqrt((1 ** 2 + 2 ** 2 + 3 ** 2 + 4 ** 2) / 4)
        rms_manual_2 = math.sqrt((1 ** 2 + 2 ** 2 + 3 ** 2 + 4 ** 2) / 4)  # те же значения по модулю

        expected_1 = x[0, 0] / rms_manual_1
        expected_2 = x[1, 0] / rms_manual_2

        # Проверяем с допустимой погрешностью
        assert torch.allclose(output[0, 0], expected_1, rtol=1e-5)
        assert torch.allclose(output[1, 0], expected_2, rtol=1e-5)

    def test_scale_effect(self):
        """Тест влияния параметра scale на выход"""
        dim = 8
        norm = RMSNorm(dim=dim)

        # Устанавливаем scale в 2.0
        norm.scale.data = 2.0 * torch.ones(dim)

        x = torch.randn(2, 5, dim)
        output = norm(x)

        # Вычисляем RMSNorm без scale
        norm_no_scale = RMSNorm(dim=dim)
        norm_no_scale.scale.data = torch.ones(dim)
        output_no_scale = norm_no_scale(x)

        # Проверяем, что output = 2 * output_no_scale
        assert torch.allclose(output, 2 * output_no_scale, rtol=1e-5)

    def test_rms_normalization_property(self):
        """Тест свойства RMS нормализации: RMS выходного тензора должно быть близко к 1"""
        norm = RMSNorm(dim=64)

        # Создаем случайные тензоры разных размеров
        test_shapes = [
            (2, 10, 64),  # batch, seq_len, dim
            (1, 20, 64),  # один элемент в batch
            (5, 1, 64),  # один элемент в последовательности
            (3, 15, 64)  # обычный случай
        ]

        for shape in test_shapes:
            x = torch.randn(*shape)
            output = norm(x)

            # Вычисляем RMS для каждого элемента последовательности в каждом batch
            rms_output = output.norm(dim=-1) / math.sqrt(shape[-1])

            # RMS должно быть примерно 1 (с учетом масштабирования параметром scale)
            # Учитываем, что scale инициализирован единицами
            assert torch.allclose(rms_output, torch.ones_like(rms_output), rtol=1e-5)

    def test_eps_protection(self, rms_norm_with_custom_eps):
        """Тест защиты от деления на ноль с помощью eps"""
        dim = 32

        # Создаем тензор с очень маленькими значениями
        x = torch.full((2, 5, dim), 1e-10)

        # Применяем RMSNorm - не должно быть NaN или inf
        output = rms_norm_with_custom_eps(x)

        # Проверяем отсутствие NaN и inf
        assert not torch.any(torch.isnan(output))
        assert not torch.any(torch.isinf(output))

    def test_gradient_flow(self, rms_norm):
        """Тест прохождения градиентов"""
        batch_size, seq_len, dim = 4, 8, 64

        # Создаем тензор с требованием градиентов
        x = torch.randn(batch_size, seq_len, dim, requires_grad=True)

        # Применяем RMSNorm
        output = rms_norm(x)

        # Создаем loss и вычисляем градиенты
        loss = output.sum()
        loss.backward()

        # Проверяем, что градиенты существуют
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Проверяем, что градиенты не все нули
        assert not torch.allclose(x.grad, torch.zeros_like(x.grad))

    def test_different_dtypes(self):
        """Тест работы с разными типами данных"""
        dim = 16

        for dtype in [torch.float32, torch.float64]:
            norm = RMSNorm(dim=dim)
            norm = norm.to(dtype)

            x = torch.randn(2, 3, dim, dtype=dtype)
            output = norm(x)

            # Проверяем сохранение типа данных
            assert output.dtype == dtype

    def test_invalid_input_dimension(self):
        """Тест обработки неверной размерности входа"""
        norm = RMSNorm(dim=32)

        # Тензор с неверной последней размерностью
        x = torch.randn(2, 10, 64)  # dim=64 вместо 32

        # Должно вызвать ошибку при вычислении нормы
        # (или работать, если последняя размерность совпадает с scale)
        # В данном случае scale имеет размер 32, а x имеет последнюю размерность 64
        # Это вызовет ошибку при умножении
        with pytest.raises(RuntimeError):
            _ = norm(x)

    def test_trainable_parameters(self):
        """Тест, что scale является обучаемым параметром"""
        norm = RMSNorm(dim=128)

        # Проверяем, что scale - это Parameter
        assert isinstance(norm.scale, nn.Parameter)

        # Проверяем количество обучаемых параметров
        params = list(norm.parameters())
        assert len(params) == 1
        assert params[0] is norm.scale


# Дополнительные тесты для граничных случаев
def test_single_element():
    """Тест для тензора с одним элементом"""
    norm = RMSNorm(dim=1)

    x = torch.tensor([[[1.0]], [[2.0]]])  # shape: [2, 1, 1]
    output = norm(x)

    # При dim=1, норма = |x| / sqrt(1) = |x|
    # Выход должен быть x / |x| = sign(x)
    expected = torch.sign(x)
    assert torch.allclose(output, expected, rtol=1e-5)


def test_large_values():
    """Тест работы с большими значениями"""
    norm = RMSNorm(dim=8)

    # Большие значения
    x = torch.tensor([[[1e6, 2e6, 3e6, 4e6, 5e6, 6e6, 7e6, 8e6]]])
    output = norm(x)

    # Проверяем отсутствие NaN/inf
    assert not torch.any(torch.isnan(output))
    assert not torch.any(torch.isinf(output))

    # Проверяем свойство RMS
    rms_output = output.norm(dim=-1) / math.sqrt(8)
    assert torch.allclose(rms_output, torch.ones_like(rms_output), rtol=1e-5)


def test_zero_scale_parameter():
    """Тест поведения при нулевом scale параметре"""
    dim = 16
    norm = RMSNorm(dim=dim)

    # Устанавливаем scale в 0
    norm.scale.data = torch.zeros(dim)

    x = torch.randn(2, 3, dim)
    output = norm(x)

    # Выход должен быть нулевым тензором той же формы
    assert torch.allclose(output, torch.zeros_like(output))
    assert output.shape == x.shape


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v"])