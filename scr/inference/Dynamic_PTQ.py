import torch
import torch.nn as nn
import numpy as np

# Словарь для хранения активаций
activations = {}
all_quantized_weights = []
def get_activation(name):
    """Функция-хук для сохранения активаций"""
    def hook(model, input, output):
        activations[name] = {
            'activation': output.detach(),
            'weights': model.weight.detach() if hasattr(model, 'weight') else None,
            'bias': model.bias.detach() if hasattr(model, 'bias') else None
        }
    return hook

# Регистрируем хуки
"""hooks = []
quant_params = {}
for i, layer in enumerate(net):

    hook = layer.register_forward_hook(get_activation(f'layer_{i}_{layer.__class__.__name__}'))
    hooks.append(hook)
output = net(x)
"""

def print_model_info(model, x):
    """Выводит полную информацию о модели: веса, смещения и активации"""

    # Собираем активации
    activations = {}
    hooks = []

    def hook_fn(name):
        def hook(module, input, output):
            activations[name] = {
                'activation': output.detach(),
                'input': input[0].detach() if input else None
            }

        return hook

    # Регистрируем хуки
    for i, layer in enumerate(model):
        hook = layer.register_forward_hook(hook_fn(f'layer_{i}'))
        hooks.append(hook)

    # Прямой проход
    output = model(x)

    # Выводим информацию
    print("=" * 80)
    print(f"Входные данные: {x.shape}")
    print("=" * 80)

    for i, (layer_name, layer) in enumerate(model.named_children()):
        print(f"\nСлой {i} ({layer.__class__.__name__}):")
        print("-" * 40)

        # Информация о параметрах слоя
        for param_name, param in layer.named_parameters():
            print(f"  {param_name}: {param.shape}")
            print(f"    Значения: mean={param.mean().item():.6f}, "
                  f"std={param.std().item():.6f}, "
                  f"range=[{param.min().item():.6f}, {param.max().item():.6f}]")
            print('параметры',param)

        # Информация об активациях
        if f'layer_{i}' in activations:
            act = activations[f'layer_{i}']['activation']
            print(f"  Активация: {act.shape}")
            print(f"    Статистика: mean={act.mean().item():.6f}, "
                  f"std={act.std().item():.6f}, "
                  f"range=[{act.min().item():.6f}, {act.max().item():.6f}]")

            # Для линейных слоев показываем пример вычислений
            if isinstance(layer, nn.Linear) and activations[f'layer_{i}']['input'] is not None:
                input_data = activations[f'layer_{i}']['input']
                print(f"    Пример вычисления (1 нейрон):")
                print(f"      input * weights[0] + bias[0] = "
                      f"sum({input_data[0]} * {layer.weight[0][:3]}...) + {layer.bias[0].item()} = "
                      f"{act[0][0].item():.4f}")

    print(f"\nФинальный выход: {output.shape}")

    # Удаляем хуки
    for hook in hooks:
        hook.remove()

    return output
# Создаем модель
"""net = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(4, 2)
)

# Тестовый вход
x = torch.randn(1, 2)
print_model_info(net,x)
# Словари для хранения
linear_activations = {}  # Только для линейных слоев
quant_params = {}
all_quantized_weights = []"""


# Функция-хук ТОЛЬКО для линейных слоев
def get_linear_activation(name):
    def hook(model, input, output):
        # Сохраняем данные только если это Linear слой
        if isinstance(model, nn.Linear):
            linear_activations[name] = {
                'layer': model,
                'activation': output.detach(),
                'weights': model.weight.detach(),
                'bias': model.bias.detach() if model.bias is not None else None,
                'input': input[0].detach() if input else None
            }

    return hook


# Регистрируем хуки ТОЛЬКО для линейных слоев
"""hooks = []
for i, layer in enumerate(net):
    if isinstance(layer, nn.Linear):  # ТОЛЬКО Linear слои
        hook = layer.register_forward_hook(
            get_linear_activation(f'linear_layer_{i}')
        )
        hooks.append(hook)
        print(f"Зарегистрирован хук для Linear слоя {i}")

# Форвард пасс
print("\nВыполняем forward pass...")
output = net(x)

# Показать какие слои были обработаны
print(f"\nОбработано линейных слоев: {len(linear_activations)}")
print(f"Всего слоев в модели: {len(net)}")

# ОБРАБАТЫВАЕМ ТОЛЬКО ЛИНЕЙНЫЕ СЛОИ
print("\n" + "=" * 50)
print("ОБРАБОТКА ЛИНЕЙНЫХ СЛОЕВ")
print("=" * 50)

for name, data in linear_activations.items():
    print(f"\n{name}:")
    print(f"  Веса shape: {data['weights'].shape}")
    print(f"  Активации shape: {data['activation'].shape}")

    if data['input'] is not None:
        print(f"  Вход shape: {data['input'].shape}")

    # Параметры квантования ДЛЯ ВЕСОВ
    w = data['weights']
    w_max = w.max()
    w_min = w.min()
    w_range = w_max - w_min

    # Избегаем деления на 0
    if w_range > 0:
        scale_w = w_range / 255.0
    else:
        scale_w = torch.tensor(1.0)

    zero_point_w = torch.round(-w_min / scale_w)

    # Сохраняем параметры квантования
    layer_idx = int(name.split('_')[-1])  # Получаем индекс из 'linear_layer_X'
    quant_params[layer_idx] = {
        'scale_w': scale_w,
        'zero_point_w': zero_point_w,
        'original_shape': w.shape
    }

    # КВАНТОВАНИЕ ВЕСОВ (весь тензор сразу)
    # 1. Квантуем в uint8
    quant_weights_uint8 = torch.round(w / scale_w + zero_point_w)
    quant_weights_uint8 = torch.clamp(quant_weights_uint8, 0, 255).to(torch.uint8)

    # 2. Конвертируем в numpy (uint8 для хранения)
    quant_weights_np = quant_weights_uint8.cpu().numpy()

    # 3. Сохраняем
    all_quantized_weights.append({
        'layer_idx': layer_idx,
        'quant_weights': quant_weights_np,  # Храним как uint8
        'original_shape': w.shape,
        'bias': data['bias']
    })

    print(f"  Scale весов: {scale_w.item():.6f}")
    print(f"  Zero point весов: {zero_point_w.item():.2f}")
    print(f"  Квантованные веса dtype: {quant_weights_np.dtype}")
    print(f"  Min/Max квант. весов: {quant_weights_np.min()}, {quant_weights_np.max()}")

# Удаляем хуки
for hook in hooks:
    hook.remove()

# ОБНОВЛЯЕМ ВЕСА ЛИНЕЙНЫХ СЛОЕВ
print("\n" + "=" * 50)
print("ОБНОВЛЕНИЕ ВЕСОВ ЛИНЕЙНЫХ СЛОЕВ")
print("=" * 50)

for quant_data in all_quantized_weights:
    layer_idx = quant_data['layer_idx']
    quant_weights_np = quant_data['quant_weights']

    # Получаем параметры квантования
    params = quant_params[layer_idx]
    scale_w = params['scale_w']
    zero_point_w = params['zero_point_w']

    print(f"\nОбработка Linear слоя {layer_idx}:")
    print(f"  Исходная форма: {quant_data['original_shape']}")
    print(f"  Квантованные веса форма: {quant_weights_np.shape}")

    # ДЕ-КВАНТОВАНИЕ
    # 1. Конвертируем в float32
    quant_weights_f32 = quant_weights_np.astype(np.float32)

    # 2. Де-квантуем
    dequant_weights_f32 = (quant_weights_f32 - zero_point_w.item()) * scale_w.item()

    # 3. Конвертируем в torch tensor
    dequant_weights_tensor = torch.from_numpy(dequant_weights_f32)

    # 4. Восстанавливаем форму если нужно
    if dequant_weights_tensor.shape != params['original_shape']:
        dequant_weights_tensor = dequant_weights_tensor.reshape(params['original_shape'])

    # НАХОДИМ И ОБНОВЛЯЕМ СООТВЕТСТВУЮЩИЙ Linear СЛОЙ
    linear_layer = None
    for i, layer in enumerate(net):
        if isinstance(layer, nn.Linear):
            # Находим линейный слой по порядковому номеру среди линейных слоев
            linear_layers = [l for l in net if isinstance(l, nn.Linear)]
            layer_position = linear_layers.index(layer)

            if layer_position == layer_idx:  # layer_idx соответствует позиции среди линейных слоев
                linear_layer = layer
                break

    if linear_layer is not None:
        with torch.no_grad():
            # Обновляем веса
            linear_layer.weight.data = dequant_weights_tensor.to(linear_layer.weight.device)

            # Сохраняем bias если есть
            if quant_data['bias'] is not None:
                linear_layer.bias.data = quant_data['bias'].to(linear_layer.bias.device)

        print(f"  Веса обновлены!")
        print(f"  Новые веса dtype: {linear_layer.weight.dtype}")
        print(f"  Новые веса shape: {linear_layer.weight.shape}")

        # Проверка ошибки
        if f'linear_layer_{layer_idx}' in linear_activations:
            original = linear_activations[f'linear_layer_{layer_idx}']['weights']
            error = torch.mean(torch.abs(original - linear_layer.weight.data))
            print(f"  Средняя ошибка восстановления: {error.item():.6f}")

# АЛЬТЕРНАТИВНЫЙ ПРОСТОЙ СПОСОБ
print("\n" + "=" * 50)
print("АЛЬТЕРНАТИВНЫЙ ПРОСТОЙ СПОСОБ")
print("=" * 50)

# Создаем новую модель для демонстрации
net_simple = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Dropout(0.0),
    nn.Linear(4, 2)
)

# Проходим по всем слоям, но обрабатываем только Linear
for i, layer in enumerate(net_simple):
    if isinstance(layer, nn.Linear):
        print(f"\nНайден Linear слой {i}")

        # Получаем веса
        w = layer.weight.data

        # Вычисляем параметры квантования
        scale = (w.max() - w.min()) / 255.0
        zero_point = torch.round(-w.min() / scale)

        # Квантуем
        quant_w = torch.round(w / scale + zero_point)
        quant_w = torch.clamp(quant_w, 0, 255).to(torch.uint8)

        # Де-квантуем (симуляция - в реальности хранили бы quant_w)
        dequant_w = (quant_w.float() - zero_point) * scale

        # Обновляем веса
        with torch.no_grad():
            layer.weight.data = dequant_w

        print(f"  Веса квантованы и обновлены")
        print(f"  Оригинальный диапазон: [{w.min():.3f}, {w.max():.3f}]")
        print(f"  Квантованный диапазон: [{quant_w.min()}, {quant_w.max()}]")

"""
# ФУНКЦИЯ ДЛЯ КВАНТОВАНИЯ ТОЛЬКО LINEAR СЛОЕВ
def quantize_linear_layers(model, dtype=torch.float32):
    """
    Квантует только Linear слои в модели

    Args:
        model: nn.Module модель
        dtype: тип данных для де-квантованных весов
    """
    quant_params = {}

    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            print(f"\nКвантование Linear слоя: {name}")

            # Получаем веса
            w = module.weight.data

            # Вычисляем параметры квантования
            if w.numel() > 0:
                scale = (w.max() - w.min()) / 255.0
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

                print(f"  Scale: {scale.item():.6f}")
                print(f"  Zero point: {zero_point.item():.2f}")

    return quant_params

"""
# Использование функции
print("\n" + "=" * 50)
print("ИСПОЛЬЗОВАНИЕ ФУНКЦИИ quantize_linear_layers")
print("=" * 50)

net_func = nn.Sequential(
    nn.Linear(2, 4),
    nn.ReLU(),
    nn.Dropout(0.1),
    nn.Linear(4, 2)
)

quant_params = quantize_linear_layers(net_func)

# Проверяем что другие слои не тронуты
print("\nТипы слоев в модели:")
for i, layer in enumerate(net_func):
    print(f"  Слой {i}: {layer.__class__.__name__}")

# Проверяем работу
print("\n" + "=" * 50)
print("ПРОВЕРКА РАБОТЫ МОДЕЛИ")
print("=" * 50)

with torch.no_grad():
    print(net_func)
    test_output = net_func(x)
    print(f"Выход модели: {test_output}")
    print(f"Форма выхода: {test_output.shape}")

x = torch.randn(1, 2)
print_model_info(net,x)"""