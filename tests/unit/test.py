def test_clean_text():
    from src.utils import clean_text

    assert isinstance(clean_text("123"), str)
    assert isinstance(clean_text(123), str)


def test_inference():
    import torch

    from src.model import RNNModel

    model = RNNModel(
        vocab_size=10,
        output_size=3,
        embed_size=10,
        hidden_size=5,
        pad_idx=0,
    ).eval()

    x = torch.tensor([0, 1, 2, 3, 4]).unsqueeze(0)
    result = model.inference(x)

    assert result.shape[0] == 1
    assert result.shape[-1] == 3


def test_get_sentiment():
    from tokenizers import Tokenizer

    from src.model import RNNModel
    from deployment.utils import get_sentiment, get_prod_model

    model, tokenizer, device = get_prod_model()

    assert isinstance(model, RNNModel)
    assert isinstance(tokenizer, Tokenizer)
    assert isinstance(get_sentiment("I love Barbie film", model, tokenizer, device), str)

    assert tokenizer.get_vocab_size() >= 5
