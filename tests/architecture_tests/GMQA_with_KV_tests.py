from GMQA_with_KV import GMQA_with_KV
import torch
import pytest
def usage():
    r = GMQA_with_KV(100, 4, 1)
    x = torch.randn(32, 32, 100)
    out = r(x)
    return out
"""u = usage()
print(u)"""


class TestGMQA_with_KV:

    @pytest.fixture
    def default_gmqa(self):
        """Базовая фикстура с типичными параметрами"""
        return GMQA_with_KV(d_model=64, num_heads=8, num_kv_groups=4)

    @pytest.fixture
    def gmqa_group_equals_heads(self):
        """GQA где каждая ключ-значение пара уникальна (MHA)"""
        return GMQA_with_KV(d_model=64, num_heads=8, num_kv_groups=8)

    @pytest.fixture
    def gmqa_single_group(self):
        """GQA где все головы используют одну пару ключ-значение (MQA)"""
        return GMQA_with_KV(d_model=64, num_heads=8, num_kv_groups=1)



    def test_initialization_success(self):

        configs = [
            (128, 8, 4),  # Стандартный GQA
            (64, 8, 8),  # MHA (группы = головы)
            (96, 6, 2),  # Другое соотношение
            (256, 16, 4),  # Большая модель
        ]

        for d_model, num_heads, num_kv_groups in configs:
            gmqa = GMQA_with_KV(d_model, num_heads, num_kv_groups)
            assert gmqa.d_model == d_model
            assert gmqa.num_heads == num_heads
            assert gmqa.num_kv_groups == num_kv_groups
            assert gmqa.d_k == d_model // num_heads

    def test_initialization_invalid_d_model(self):

        with pytest.raises(AssertionError, match="d_model должно делиться на num_heads"):
            GMQA_with_KV(d_model=65, num_heads=8, num_kv_groups=4)

    def test_initialization_invalid_kv_groups(self):

        with pytest.raises(AssertionError, match="num_heads должно делиться на num_kv_groups"):
            GMQA_with_KV(d_model=64, num_heads=8, num_kv_groups=3)

    def test_parameter_sizes(self):
        """Тест размеров обучаемых параметров"""
        gmqa = GMQA_with_KV(d_model=64, num_heads=8, num_kv_groups=4)

        # Проверяем размеры матриц проекции
        assert gmqa.W_Q.weight.shape == (64, 64)
        assert gmqa.W_K.weight.shape == (32, 64)  # d_model * num_kv_groups / num_heads = 64 * 4 / 8 = 32
        assert gmqa.W_V.weight.shape == (32, 64)
        assert gmqa.W_O.weight.shape == (64, 64)



    def test_forward_output_shapes(self, default_gmqa):
        """Тест размеров выходных тензоров"""
        batch_size, seq_len = 4, 10
        x = torch.randn(batch_size, seq_len, 64)

        output, attn_weights, present_kv = default_gmqa(x)

        # Проверяем размеры выхода
        assert output.shape == (batch_size, seq_len, 64)
        assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)  # (B, H, T, T)
        assert present_kv is None  # use_cache=False по умолчанию

    def test_forward_with_cache(self, default_gmqa):

        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, 64)

        # Первый вызов с кэшированием
        output1, attn1, kv1 = default_gmqa(x, use_cache=True)
        assert kv1 is not None
        assert len(kv1) == 2
        k1, v1 = kv1
        assert k1.shape == (batch_size, 4, seq_len, 8)  # (B, G, T, D)
        assert v1.shape == (batch_size, 4, seq_len, 8)

        # Второй вызов с передачей прошлого кэша
        x2 = torch.randn(batch_size, 3, 64)  # Новая последовательность
        output2, attn2, kv2 = default_gmqa(x2, use_cache=True, past_kv=kv1)

        # Проверяем, что кэш увеличился
        k2, v2 = kv2
        assert k2.shape == (batch_size, 4, seq_len + 3, 8)
        assert v2.shape == (batch_size, 4, seq_len + 3, 8)

    def test_attention_weights_properties(self, default_gmqa):
        """Тест свойств матрицы внимания"""
        batch_size, seq_len = 3, 7
        x = torch.randn(batch_size, seq_len, 64)

        _, attn_weights, _ = default_gmqa(x)

        # Проверяем, что сумма по последнему измерению равна 1 (softmax)
        sums = attn_weights.sum(dim=-1)
        assert torch.allclose(sums, torch.ones_like(sums), rtol=1e-5)

        # Проверяем, что все значения в диапазоне [0, 1]
        assert torch.all(attn_weights >= 0)
        assert torch.all(attn_weights <= 1)



    def test_multi_head_attention_mode(self, gmqa_group_equals_heads):
        """Тест режима Multi-Head Attention (num_kv_groups = num_heads)"""
        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, seq_len, 64)

        output, attn_weights, _ = gmqa_group_equals_heads(x)

        # Проверяем размеры
        assert output.shape == (batch_size, seq_len, 64)
        assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)

    def test_multi_query_attention_mode(self, gmqa_single_group):
        """Тест режима Multi-Query Attention (num_kv_groups = 1)"""
        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, seq_len, 64)

        output, attn_weights, present_kv = gmqa_single_group(x, use_cache=True)

        # Проверяем размеры
        assert output.shape == (batch_size, seq_len, 64)
        assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)

        # Проверяем, что K/V имеют только одну группу
        k, v = present_kv
        assert k.shape == (batch_size, 1, seq_len, 8)  # G=1
        assert v.shape == (batch_size, 1, seq_len, 8)

    def test_different_batch_sizes(self, default_gmqa):
        """Тест работы с разными размерами батча"""
        test_shapes = [
            (1, 5, 64),  # batch_size=1
            (4, 10, 64),  # средний батч
            (16, 3, 64),  # большой батч, короткие последовательности
        ]

        for batch_size, seq_len, d_model in test_shapes:
            x = torch.randn(batch_size, seq_len, d_model)
            output, attn_weights, _ = default_gmqa(x)

            assert output.shape == (batch_size, seq_len, d_model)
            assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)

    def test_different_sequence_lengths(self, default_gmqa):
        """Тест работы с разными длинами последовательностей"""
        batch_size = 2
        seq_lengths = [1, 5, 20, 50]

        for seq_len in seq_lengths:
            x = torch.randn(batch_size, seq_len, 64)
            output, attn_weights, _ = default_gmqa(x)

            assert output.shape == (batch_size, seq_len, 64)
            assert attn_weights.shape == (batch_size, 8, seq_len, seq_len)



    def test_cache_consistency(self, default_gmqa):
        """Тест согласованности с кэшированием и без"""
        batch_size, seq_len = 3, 8
        x = torch.randn(batch_size, seq_len, 64)

        # Без кэша
        output_no_cache, attn_no_cache, _ = default_gmqa(x, use_cache=False)

        # С кэшом (но без past_kv)
        output_cache, attn_cache, kv_cache = default_gmqa(x, use_cache=True)

        # Должны быть одинаковыми
        assert torch.allclose(output_no_cache, output_cache, rtol=1e-5)
        assert torch.allclose(attn_no_cache, attn_cache, rtol=1e-5)
        assert kv_cache is not None

    def test_incremental_decoding(self, default_gmqa):
        """Тест инкрементального декодирования (токен за токеном)"""
        batch_size = 2
        d_model = 64

        # Эмулируем генерацию токенов по одному
        past_kv = None
        all_outputs = []

        for step in range(5):
            # Один токен на шаг
            x_step = torch.randn(batch_size, 1, d_model)

            # Forward с прошлым кэшем
            output_step, _, past_kv = default_gmqa(
                x_step,
                use_cache=True,
                past_kv=past_kv
            )

            all_outputs.append(output_step)

        # Проверяем, что кэш рос с каждым шагом
        k, v = past_kv
        assert k.shape == (batch_size, 4, 5, 8)  # G=4, total_seq_len=5
        assert v.shape == (batch_size, 4, 5, 8)

    def test_gradient_flow(self, default_gmqa):
        """Тест прохождения градиентов"""
        batch_size, seq_len = 2, 7
        x = torch.randn(batch_size, seq_len, 64, requires_grad=True)

        output, attn_weights, _ = default_gmqa(x)

        # Создаем loss и вычисляем градиенты
        loss = output.sum() + attn_weights.sum()
        loss.backward()

        # Проверяем, что градиенты существуют
        assert x.grad is not None
        assert x.grad.shape == x.shape

        # Проверяем градиенты параметров
        for name, param in default_gmqa.named_parameters():
            assert param.grad is not None, f"Нет градиента для {name}"
            assert not torch.allclose(param.grad, torch.zeros_like(param.grad))

    def test_training_mode(self, default_gmqa):
        """Тест работы в режиме обучения"""
        batch_size, seq_len = 4, 12
        x = torch.randn(batch_size, seq_len, 64)

        # В режиме обучения
        default_gmqa.train()
        output_train, attn_train, _ = default_gmqa(x)

        # В режиме инференса
        default_gmqa.eval()
        output_eval, attn_eval, _ = default_gmqa(x)

        # Должны быть одинаковыми (нет dropout/batch norm)
        assert torch.allclose(output_train, output_eval, rtol=1e-5)
        assert torch.allclose(attn_train, attn_eval, rtol=1e-5)

    def test_single_token(self, default_gmqa):
        """Тест с одним токеном"""
        x = torch.randn(1, 1, 64)  # batch=1, seq_len=1
        output, attn_weights, _ = default_gmqa(x)

        assert output.shape == (1, 1, 64)
        assert attn_weights.shape == (1, 8, 1, 1)

        # Матрица внимания для одного токена должна быть [[1]]
        assert torch.allclose(attn_weights, torch.ones_like(attn_weights))

    def test_large_tensors(self):
        """Тест работы с большими тензорами (память)"""
        # Используем меньшую модель для экономии памяти
        gmqa = GMQA_with_KV(d_model=32, num_heads=4, num_kv_groups=2)

        batch_size, seq_len = 8, 512  # Большая последовательность
        x = torch.randn(batch_size, seq_len, 32)

        # Должно работать без ошибок памяти
        output, attn_weights, _ = gmqa(x)

        assert output.shape == (batch_size, seq_len, 32)
        assert attn_weights.shape == (batch_size, 4, seq_len, seq_len)

    def test_nan_protection(self, default_gmqa):
        """Тест защиты от NaN/Inf значений"""
        batch_size, seq_len = 2, 5

        # Тестируем с очень большими значениями
        x_large = torch.full((batch_size, seq_len, 64), 1e10)
        output_large, _, _ = default_gmqa(x_large)
        assert not torch.any(torch.isnan(output_large))
        assert not torch.any(torch.isinf(output_large))

        # Тестируем с очень маленькими значениями
        x_small = torch.full((batch_size, seq_len, 64), 1e-10)
        output_small, _, _ = default_gmqa(x_small)
        assert not torch.any(torch.isnan(output_small))
        assert not torch.any(torch.isinf(output_small))


    def test_wrong_input_dimension(self, default_gmqa):
        """Тест обработки неверной размерности входа"""
        # Неверная последняя размерность
        x_wrong = torch.randn(2, 10, 32)  # должно быть 64

        with pytest.raises(RuntimeError):
            _ = default_gmqa(x_wrong)

    def test_invalid_past_kv_structure(self, default_gmqa):
        """Тест передачи некорректного past_kv"""
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, 64)

        # Неправильная структура past_kv
        invalid_past_kv = (torch.randn(2, 4, 3, 8),)  # только K, нет V

        with pytest.raises((RuntimeError, ValueError, TypeError)):
            _ = default_gmqa(x, past_kv=invalid_past_kv)

    def test_past_kv_wrong_shape(self, default_gmqa):
        """Тест past_kv с неверной формой"""
        batch_size, seq_len = 2, 5
        x = torch.randn(batch_size, seq_len, 64)

        # Past_kv с неверным числом групп
        wrong_kv = (
            torch.randn(batch_size, 8, 10, 8),  # G=8, должно быть 4
            torch.randn(batch_size, 8, 10, 8)
        )

        with pytest.raises(RuntimeError):
            _ = default_gmqa(x, past_kv=wrong_kv)



    def test_output_range(self, default_gmqa):
        """Тест, что выход имеет разумный диапазон значений"""
        batch_size, seq_len = 3, 8
        x = torch.randn(batch_size, seq_len, 64)

        output, _, _ = default_gmqa(x)

        # Проверяем, что выход не все zeros/NaNs
        assert not torch.allclose(output, torch.zeros_like(output))
        assert not torch.any(torch.isnan(output))

        # Проверяем разумный std (не слишком большой/маленький)
        output_std = output.std().item()
        assert 0.1 < output_std < 10.0, f"Слишком экстремальный std: {output_std}"

    def test_deterministic_behavior(self, default_gmqa):
        """Тест детерминированности forward pass"""
        batch_size, seq_len = 2, 6
        x = torch.randn(batch_size, seq_len, 64)

        # Два вызова с одинаковым входом
        output1, attn1, _ = default_gmqa(x)
        output2, attn2, _ = default_gmqa(x)

        # Должны быть идентичны (если нет случайности)
        assert torch.allclose(output1, output2, rtol=1e-7)
        assert torch.allclose(attn1, attn2, rtol=1e-7)



class TestGMQAIntegration:

    def test_with_autograd_graph(self):
        """Тест создания графа вычислений"""
        gmqa = GMQA_with_KV(d_model=32, num_heads=4, num_kv_groups=2)

        x = torch.randn(2, 10, 32, requires_grad=True)

        # Двойной forward для проверки графа
        output1, _, kv1 = gmqa(x[:1, :5], use_cache=True)
        output2, _, kv2 = gmqa(x[:1, 5:], use_cache=True, past_kv=kv1)

        # Собираем все выходы
        total_output = torch.cat([output1, output2], dim=1)

        # Backward
        loss = total_output.sum()
        loss.backward()

        # Проверяем градиенты
        assert x.grad is not None
        assert x.grad.shape == x.shape

    def test_model_serialization(self, tmp_path):
        """Тест сериализации и десериализации модели"""
        gmqa = GMQA_with_KV(d_model=64, num_heads=8, num_kv_groups=4)

        # Сохраняем
        model_path = tmp_path / "gmqa_model.pth"
        torch.save(gmqa.state_dict(), model_path)

        # Загружаем в новую модель
        gmqa_loaded = GMQA_with_KV(d_model=64, num_heads=8, num_kv_groups=4)
        gmqa_loaded.load_state_dict(torch.load(model_path))

        # Проверяем, что работает одинаково
        x = torch.randn(2, 5, 64)
        output1, attn1, _ = gmqa(x)
        output2, attn2, _ = gmqa_loaded(x)

        assert torch.allclose(output1, output2, rtol=1e-5)
        assert torch.allclose(attn1, attn2, rtol=1e-5)


if __name__ == "__main__":
    # Запуск тестов
    pytest.main([__file__, "-v", "--tb=short"])