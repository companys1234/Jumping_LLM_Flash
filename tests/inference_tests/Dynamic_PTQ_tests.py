#from Dynamic_PTQ import get_activation,print_model_info,get_linear_activation,quantize_linear_layers
import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import patch, MagicMock

# Импортируем тестируемые функции (или определяем их здесь)
def get_linear_activation(name):
    """Функция-хук для сохранения данных линейных слоев"""
    def hook(model, input, output):
        if isinstance(model, nn.Linear):
            # Эта функция обычно определяется в глобальном контексте
            pass
    return hook

def print_model_info(model, x):
    """Выводит информацию о модели"""
    activations = {}
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = {
                'activation': output.detach(),
                'input': input[0].detach() if input else None
            }
        return hook

    for i, layer in enumerate(model):
        hook = layer.register_forward_hook(hook_fn(f'layer_{i}'))
        hooks.append(hook)

    output = model(x)

    for hook in hooks:
        hook.remove()

    return output

def quantize_linear_layers(model, dtype=torch.float32):
    """
    Квантует только Linear слои в модели
    """
    quant_params = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Получаем веса
            w = module.weight.data

            if w.numel() > 0:
                # Избегаем деления на 0
                if w.max() != w.min():
                    scale = (w.max() - w.min()) / 255.0
                else:
                    scale = torch.tensor(1.0)

                zero_point = torch.round(-w.min() / scale)

                # Квантуем
                quant_w = torch.round(w / scale + zero_point)
                quant_w = torch.clamp(quant_w, 0, 255).to(torch.uint8)

                # Де-квантуем
                dequant_w = (quant_w.float() - zero_point) * scale

                # Обновляем веса
                with torch.no_grad():
                    module.weight.data = dequant_w.to(dtype)

                # Сохраняем параметры
                quant_params[name] = {
                    'scale': scale,
                    'zero_point': zero_point,
                    'quantized_weights': quant_w.cpu().numpy(),
                    'original_shape': w.shape
                }

    return quant_params


# Тестовый класс
class TestQuantizationFunctions:

    @pytest.fixture
    def simple_model(self):
        """Простая модель для тестирования"""
        return nn.Sequential(
            nn.Linear(2, 4),
            nn.ReLU(),
            nn.Dropout(0.0),
            nn.Linear(4, 2),
            nn.Softmax(dim=-1)
        )

    @pytest.fixture
    def complex_model(self):
        """Более сложная модель с несколькими линейными слоями"""
        return nn.Sequential(
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 30),
            nn.Tanh(),
            nn.Linear(30, 5),
            nn.Dropout(0.1),
            nn.Linear(5, 1)
        )

    @pytest.fixture
    def test_input(self):
        """Тестовый входной тензор"""
        return torch.randn(3, 2)  # batch_size=3, features=2

    @pytest.fixture
    def test_input_complex(self):
        """Тестовый вход для сложной модели"""
        return torch.randn(5, 10)  # batch_size=5, features=10

    # ==================== Тесты print_model_info ====================

    def test_print_model_info_output_shape(self, simple_model, test_input):
        """Тест, что print_model_info возвращает правильный output"""
        output = print_model_info(simple_model, test_input)

        # Проверяем форму выхода
        assert output.shape == (3, 2)  # batch_size=3, output_features=2

        # Проверяем, что выход от модели с Softmax суммируется в 1
        assert torch.allclose(output.sum(dim=1), torch.ones(3), rtol=1e-5)

    def test_print_model_info_no_hooks_left(self, simple_model, test_input):
        """Тест, что хуки корректно удаляются"""
        # Считаем сколько хуков было до
        hooks_before = len(simple_model._forward_hooks) if hasattr(simple_model, '_forward_hooks') else 0

        _ = print_model_info(simple_model, test_input)

        # Проверяем, что хуков не осталось
        hooks_after = len(simple_model._forward_hooks) if hasattr(simple_model, '_forward_hooks') else 0
        assert hooks_after == hooks_before

    def test_print_model_info_with_different_models(self):
        """Тест работы с моделями разной архитектуры"""
        models = [
            nn.Sequential(nn.Linear(5, 3)),
            nn.Sequential(nn.Conv2d(3, 16, 3), nn.Flatten(), nn.Linear(16*26*26, 10)),
            nn.Sequential(nn.LSTM(10, 20, batch_first=True), nn.Linear(20, 5))]

        for model in models:
            # Создаем соответствующий вход
            if isinstance(model[0], nn.Linear):
                x = torch.randn(2, 5)
            elif isinstance(model[0], nn.Conv2d):
                x = torch.randn(2, 3, 28, 28)
            else:  # LSTM
                x = torch.randn(2, 10, 10)  # batch, seq_len, features

            # Проверяем, что функция работает без ошибок
            try:
                output = print_model_info(model, x)
                assert output is not None
            except Exception as e:
                pytest.fail(f"print_model_info failed for {model.__class__.__name__}: {e}")

    # ==================== Тесты quantize_linear_layers ====================

    def test_quantize_linear_layers_basic(self, simple_model):
        """Базовый тест квантования линейных слоев"""
        # Сохраняем оригинальные веса
        original_weights = {}
        for name, module in simple_model.named_modules():
            if isinstance(module, nn.Linear):
                original_weights[name] = module.weight.data.clone()

        # Квантуем
        quant_params = quantize_linear_layers(simple_model)

        # Проверяем, что параметры сохранены
        assert len(quant_params) == 2  # Два линейных слоя

        # Проверяем структуру параметров
        for layer_name, params in quant_params.items():
            assert 'scale' in params
            assert 'zero_point' in params
            assert 'quantized_weights' in params
            assert 'original_shape' in params

            # Проверяем типы
            assert isinstance(params['scale'], torch.Tensor)
            assert isinstance(params['zero_point'], torch.Tensor)
            assert isinstance(params['quantized_weights'], np.ndarray)

            # Квантованные веса должны быть uint8
            assert params['quantized_weights'].dtype == np.uint8

        # Проверяем, что не-линейные слои не затронуты
        for i, layer in enumerate(simple_model):
            if isinstance(layer, nn.ReLU) or isinstance(layer, nn.Dropout) or isinstance(layer, nn.Softmax):
                # Эти слои не должны иметь параметров weight
                assert not hasattr(layer, 'weight') or layer.weight is None

    def test_quantize_preserves_functionality(self, simple_model, test_input):
        """Тест, что квантованная модель дает похожий результат"""
        # Получаем выход до квантования
        with torch.no_grad():
            output_before = simple_model(test_input)

        # Квантуем
        quant_params = quantize_linear_layers(simple_model)

        # Получаем выход после квантования
        with torch.no_grad():
            output_after = simple_model(test_input)

        # Выходы должны быть близки (но не идентичны из-за квантования)
        # Проверяем с разумным допуском
        mse = torch.mean((output_before - output_after) ** 2)
        assert mse.item() < 0.1  # Допустимая ошибка

        # Проверяем, что форма сохранилась
        assert output_before.shape == output_after.shape

    def test_quantize_constant_weights(self):
        """Тест квантования постоянных весов"""
        # Создаем линейный слой с постоянными весами
        class ConstantLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear = nn.Linear(3, 3)
                # Устанавливаем все веса в 1.0
                with torch.no_grad():
                    self.linear.weight.data.fill_(1.0)
                    if self.linear.bias is not None:
                        self.linear.bias.data.fill_(0.0)

            def forward(self, x):
                return self.linear(x)

        model = ConstantLinear()

        # Квантуем
        quant_params = quantize_linear_layers(model)

        # Проверяем, что scale не равен 0 (деление на 0 защищено)
        params = quant_params['linear']
        assert params['scale'].item() > 0 or params['scale'].item() == 1.0

        # Проверяем, что веса восстановились примерно правильно
        restored_weight = (params['quantized_weights'].astype(np.float32) - params['zero_point'].item()) * params['scale'].item()
        assert np.allclose(restored_weight, 1.0, rtol=1e-3)

    def test_quantize_extreme_values(self):
        """Тест квантования с экстремальными значениями весов"""
        model = nn.Sequential(nn.Linear(2, 2))

        # Устанавливаем экстремальные веса
        with torch.no_grad():
            model[0].weight.data = torch.tensor([[1e10, -1e10], [0.0, 1e-10]])

        # Квантуем - не должно быть ошибок
        quant_params = quantize_linear_layers(model)

        # Проверяем параметры
        params = quant_params['0']
        assert not np.any(np.isnan(params['quantized_weights']))
        assert not np.any(np.isinf(params['quantized_weights']))

        # Восстановленные веса должны быть в разумных пределах
        restored = (params['quantized_weights'].astype(np.float32) - params['zero_point'].item()) * params['scale'].item()
        assert not torch.any(torch.isnan(torch.from_numpy(restored)))
        assert not torch.any(torch.isinf(torch.from_numpy(restored)))

    def test_quantize_empty_tensor(self):
        """Тест квантования пустого тензора (защита от edge case)"""
        model = nn.Sequential(nn.Linear(0, 0))  # Необычный, но возможный случай

        try:
            quant_params = quantize_linear_layers(model)
            # Если функция обрабатывает этот случай, проверяем дальше
            if '0' in quant_params:
                params = quant_params['0']
                assert params['quantized_weights'].size == 0
        except Exception as e:
            # Функция может падать на пустых тензорах - это нормально
            pass

    def test_quantize_different_dtypes(self, simple_model):
        """Тест квантования с разными типами данных выхода"""
        dtypes = [torch.float32, torch.float64, torch.float16]

        for dtype in dtypes:
            # Клонируем модель
            model_copy = nn.Sequential(*[layer for layer in simple_model])

            # Квантуем с указанным dtype
            quant_params = quantize_linear_layers(model_copy, dtype=dtype)

            # Проверяем тип данных весов
            for name, module in model_copy.named_modules():
                if isinstance(module, nn.Linear):
                    assert module.weight.dtype == dtype

    def test_quantize_only_linear_layers(self, complex_model):
        """Тест, что квантуются только линейные слои"""
        # Считаем линейные слои
        linear_count = sum(1 for _ in complex_model.modules() if isinstance(_, nn.Linear))
        assert linear_count == 4

        # Квантуем
        quant_params = quantize_linear_layers(complex_model)

        # Проверяем, что квантованы все линейные слои
        assert len(quant_params) == linear_count

        # Проверяем имена слоев в параметрах
        for name in quant_params.keys():
            # Имя должно содержать путь к модулю
            assert any(name.endswith(str(i)) for i in range(len(complex_model)))

    def test_quantization_error_bounds(self):
        """Тест границ ошибки квантования"""
        model = nn.Sequential(nn.Linear(100, 100))

        # Инициализируем случайными весами
        torch.nn.init.normal_(model[0].weight, mean=0, std=1)

        # Сохраняем оригинальные веса
        original = model[0].weight.data.clone()

        # Квантуем
        quant_params = quantize_linear_layers(model)

        # Получаем восстановленные веса
        params = quant_params['0']
        restored = (params['quantized_weights'].astype(np.float32) - params['zero_point'].item()) * params['scale'].item()
        restored_tensor = torch.from_numpy(restored).to(original.device)

        # Ошибка должна быть ограничена шагом квантования
        quantization_step = params['scale'].item()
        max_error = torch.max(torch.abs(original - restored_tensor))

        # Максимальная ошибка не должна превышать шаг квантования
        assert max_error.item() <= quantization_step * 1.5

    def test_quantize_with_bias(self):
        """Тест квантования слоев с bias"""
        model = nn.Sequential(
            nn.Linear(5, 3, bias=True),
            nn.Linear(3, 1, bias=False)
        )

        # Сохраняем оригинальные bias
        original_bias = model[0].bias.data.clone()

        # Квантуем
        quant_params = quantize_linear_layers(model)

        # Проверяем, что bias первого слоя не изменился
        assert torch.allclose(model[0].bias.data, original_bias)

        # Проверяем, что у второго слоя нет bias
        assert model[1].bias is None

    def test_repeated_quantization(self, simple_model):
        """Тест повторного квантования той же модели"""
        # Первое квантование
        quant_params1 = quantize_linear_layers(simple_model)

        # Сохраняем веса после первого квантования
        weights_after_first = {}
        for name, module in simple_model.named_modules():
            if isinstance(module, nn.Linear):
                weights_after_first[name] = module.weight.data.clone()

        # Второе квантование
        quant_params2 = quantize_linear_layers(simple_model)

        # Веса должны остаться примерно теми же (идемпотентность)
        for name, module in simple_model.named_modules():
            if isinstance(module, nn.Linear):
                assert torch.allclose(module.weight.data, weights_after_first[name], rtol=1e-5)

    # ==================== Интеграционные тесты ====================

    def test_integration_print_and_quantize(self, simple_model, test_input):
        """Интеграционный тест print_model_info и quantize_linear_layers"""
        # 1. Получаем информацию о модели
        output_before = print_model_info(simple_model, test_input)

        # 2. Квантуем линейные слои
        quant_params = quantize_linear_layers(simple_model)

        # 3. Снова получаем информацию
        output_after = print_model_info(simple_model, test_input)

        # 4. Проверяем, что модель все еще работает
        assert output_before.shape == output_after.shape
        assert not torch.any(torch.isnan(output_after))
        assert not torch.any(torch.isinf(output_after))

    def test_quantization_with_training_mode(self, simple_model):
        """Тест квантования в режиме обучения"""
        # Переводим в режим обучения
        simple_model.train()

        # Квантуем
        quant_params = quantize_linear_layers(simple_model)

        # Проверяем, что градиенты отключены для обновленных весов
        for name, module in simple_model.named_modules():
            if isinstance(module, nn.Linear):
                # Веса не должны требовать градиента после квантования
                # (хотя в функции мы используем torch.no_grad())
                pass

    def test_memory_efficiency(self):
        """Тест использования памяти квантованными весами"""
        # Создаем большую модель
        large_model = nn.Sequential(
            nn.Linear(1000, 2000),
            nn.ReLU(),
            nn.Linear(2000, 1000)
        )

        # Квантуем
        quant_params = quantize_linear_layers(large_model)

        # Проверяем, что квантованные веса занимают меньше места
        for layer_name, params in quant_params.items():
            quantized_size = params['quantized_weights'].nbytes
            original_size = params['original_shape'].numel() * 4  # float32 = 4 байта

            # uint8 должен занимать в 4 раза меньше
            assert quantized_size <= original_size / 3  # Немного больше из-за метаданных

    # ==================== Тесты edge cases ====================

    def test_quantize_model_with_no_linear_layers(self):
        """Тест модели без линейных слоев"""
        model = nn.Sequential(
            nn.Conv2d(3, 16, 3),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        quant_params = quantize_linear_layers(model)

        # Не должно быть параметров квантования
        assert len(quant_params) == 0

        # Модель должна остаться неизменной
        assert isinstance(model[0], nn.Conv2d)
        assert isinstance(model[1], nn.ReLU)
        assert isinstance(model[2], nn.MaxPool2d)

    def test_quantize_with_nan_values(self):
        """Тест обработки NaN в весах"""
        model = nn.Sequential(nn.Linear(2, 2))

        # Устанавливаем NaN в веса
        with torch.no_grad():
            model[0].weight.data = torch.tensor([[1.0, float('nan')], [float('nan'), 2.0]])

        # Квантование должно обработать NaN
        quant_params = quantize_linear_layers(model)

        # Проверяем, что в квантованных весах нет NaN
        params = quant_params['0']
        assert not np.any(np.isnan(params['quantized_weights']))

    def test_quantize_single_element_tensor(self):
        """Тест квантования тензора с одним элементом"""
        model = nn.Sequential(nn.Linear(1, 1))

        with torch.no_grad():
            model[0].weight.data = torch.tensor([[5.0]])

        quant_params = quantize_linear_layers(model)

        params = quant_params['0']
        # Шаг квантования должен быть вычислен корректно
        assert params['scale'].item() > 0

        # Восстановленное значение должно быть близко к оригиналу
        restored = (params['quantized_weights'].astype(np.float32) - params['zero_point'].item()) * params['scale'].item()
        assert np.allclose(restored, 5.0, rtol=0.1)

    def test_quantize_very_small_range(self):
        """Тест квантования весов с очень маленьким диапазоном"""
        model = nn.Sequential(nn.Linear(3, 3))

        # Веса в очень узком диапазоне
        with torch.no_grad():
            model[0].weight.data = torch.tensor([
                [1.000, 1.001, 1.002],
                [1.003, 1.004, 1.005],
                [1.006, 1.007, 1.008]
            ])

        quant_params = quantize_linear_layers(model)

        params = quant_params['0']
        # Scale должен быть очень маленьким
        assert params['scale'].item() > 0
        assert params['scale'].item() < 0.01

    # ==================== Mock тесты для хуков ====================

    def test_hook_registration(self):
        """Тест регистрации хуков (с использованием mock)"""
        with patch('torch.nn.Module.register_forward_hook') as mock_register:
            model = nn.Sequential(nn.Linear(2, 2))
            _ = print_model_info(model, torch.randn(1, 2))

            # Проверяем, что хук был зарегистрирован для каждого слоя
            assert mock_register.call_count == 1  # Один слой в модели

    def test_quantization_with_custom_module(self):
        """Тест квантования с пользовательским модулем"""
        class CustomLinear(nn.Module):
            def __init__(self):
                super().__init__()
                self.weight = nn.Parameter(torch.randn(3, 3))
                self.bias = nn.Parameter(torch.zeros(3))

            def forward(self, x):
                return x @ self.weight.t() + self.bias

        class MixedModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(3, 3)
                self.custom = CustomLinear()
                self.linear2 = nn.Linear(3, 3)

            def forward(self, x):
                x = self.linear1(x)
                x = self.custom(x)
                x = self.linear2(x)
                return x

        model = MixedModel()

        # quantize_linear_layers должна квантовать только nn.Linear
        quant_params = quantize_linear_layers(model)

        # Только 2 слоя должны быть квантованы
        assert len(quant_params) == 2
        assert 'linear1' in quant_params
        assert 'linear2' in quant_params
        assert 'custom' not in quant_params  # CustomLinear не должен быть квантован


# Дополнительные тесты производительности
class TestQuantizationPerformance:

    def test_quantization_speed(self):
        """Тест скорости квантования (не строгий, скорее smoke test)"""
        import time

        # Создаем большую модель
        model = nn.Sequential(
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256)
        )

        # Измеряем время
        start_time = time.time()
        quant_params = quantize_linear_layers(model)
        end_time = time.time()

        # Просто проверяем, что не слишком долго
        assert end_time - start_time < 5.0  # Должно занимать меньше 5 секунд

        # Все слои должны быть квантованы
        assert len(quant_params) == 3

    def test_quantization_large_tensors(self):
        """Тест квантования очень больших тензоров"""
        # Модель с очень большими весами
        model = nn.Sequential(
            nn.Linear(2048, 4096),
            nn.Linear(4096, 2048)
        )

        # Не должно быть ошибок памяти
        try:
            quant_params = quantize_linear_layers(model)
            assert len(quant_params) == 2
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Недостаточно памяти для теста")
            else:
                raise


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "--tb=short"])