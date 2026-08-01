"""
Root cause analysis: Why does Spectral fail on RoBERTa CoLA?

Hypothesis: Spectral's effective output magnitude (scaling=1.0) is ~20x weaker
than FourierFT's (scaling=150). On BERT, the tiny classifier needed the adapter.
On RoBERTa, the 592K-param 2-layer MLP classifier can solve CoLA from frozen
features alone, ignoring the weak adapter signal.

This script verifies by measuring actual adapter output magnitudes on both models.
"""
import os, sys, math
import torch
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), 'src'))
from spectral_adapter import get_spectral_adapter_model, SpectralAdapterLinear

os.environ['HF_HOME'] = os.environ.get('HF_HOME', './data')
os.environ['HF_DATASETS_CACHE'] = os.environ.get('HF_DATASETS_CACHE', './data')
os.environ['TRANSFORMERS_CACHE'] = os.environ.get('TRANSFORMERS_CACHE', './data')

def measure_adapter_magnitudes(model_name, device='cpu'):
    """Measure adapter and base output magnitudes for Spectral and FourierFT."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import FourierFTConfig, get_peft_model

    print(f"\n{'='*70}")
    print(f"MODEL: {model_name}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./data')

    # Fixed test input
    texts = ["The quick brown fox jumps over the lazy dog.",
             "This is a grammatically correct sentence.",
             "Him go store yesterday for buy milk."]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    enc['labels'] = torch.tensor([1, 1, 0])
    batch = {k: v.to(device) for k, v in enc.items()}

    # --- SPECTRAL ---
    for scaling in [1.0, 10.0, 20.0, 50.0, 100.0, 150.0]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, cache_dir='./data'
        )

        # Determine target modules
        if 'roberta' in model_name or 'bert' in model_name:
            targets = ['query', 'value']
        else:
            targets = ['q_proj', 'v_proj']

        model = get_spectral_adapter_model(
            model, target_modules=targets,
            p=16, q=16, scaling=scaling, d_initial=0.01,
            freq_mode='contiguous'
        )
        model = model.to(device).eval()

        # Hook to capture adapter delta
        deltas = {}
        bases = {}
        def make_hook(name):
            def fn(module, inp, out):
                with torch.no_grad():
                    x = inp[0]
                    base = module.base_layer(x)
                    deltas[name] = (out - base).detach()
                    bases[name] = base.detach()
            return fn

        hooks = []
        for name, mod in model.named_modules():
            if isinstance(mod, SpectralAdapterLinear):
                hooks.append(mod.register_forward_hook(make_hook(name)))

        with torch.no_grad():
            outputs = model(**batch)

        # Summarize
        delta_rms_list = []
        base_rms_list = []
        for name in sorted(deltas.keys()):
            d = deltas[name]
            b = bases[name]
            delta_rms_list.append(d.pow(2).mean().sqrt().item())
            base_rms_list.append(b.pow(2).mean().sqrt().item())

        avg_delta = np.mean(delta_rms_list)
        avg_base = np.mean(base_rms_list)
        avg_ratio = avg_delta / (avg_base + 1e-12)

        print(f"  SPECTRAL s={scaling:6.1f}: avg_delta_rms={avg_delta:.6e}, avg_base_rms={avg_base:.6e}, ratio={avg_ratio:.6e}")

        for h in hooks:
            h.remove()
        del model

    # --- FOURIERFT ---
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, cache_dir='./data'
    )
    config = FourierFTConfig(
        target_modules=targets,
        n_frequency=256,
        scaling=150.0,
        random_loc_seed=42,
    )
    model = get_peft_model(model, config)
    model = model.to(device).eval()

    # For FourierFT, measure by running with and without adapter
    # Enable adapter
    with torch.no_grad():
        out_with = model(**batch)
        logits_with = out_with.logits.clone()

    # Disable adapter (set scaling to 0 by disabling)
    model.disable_adapter_layers()
    with torch.no_grad():
        out_without = model(**batch)
        logits_without = out_without.logits.clone()
    model.enable_adapter_layers()

    logit_diff = (logits_with - logits_without).pow(2).mean().sqrt().item()
    print(f"  FOURIERFT s=150.0: logit_diff_rms={logit_diff:.6e} (adapter contribution to final logits)")

    del model

    # --- Classifier analysis ---
    print(f"\n  CLASSIFIER ANALYSIS ({model_name}):")
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, cache_dir='./data'
    ).to(device)
    classifier_params = sum(p.numel() for n, p in model.named_parameters()
                          if 'classifier' in n or 'score' in n)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Classifier params: {classifier_params:,} ({classifier_params/total_params*100:.2f}% of total)")

    # Check classifier architecture
    for name, module in model.named_modules():
        if 'classifier' in name or 'score' in name:
            if hasattr(module, 'weight'):
                print(f"    {name}: {module.__class__.__name__}({module.in_features}->{module.out_features})")
    del model


def gradient_analysis(model_name, device='cpu'):
    """Compare gradient magnitudes between Spectral and FourierFT."""
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from peft import FourierFTConfig, get_peft_model

    print(f"\n{'='*70}")
    print(f"GRADIENT ANALYSIS: {model_name}")
    print(f"{'='*70}")

    tokenizer = AutoTokenizer.from_pretrained(model_name, cache_dir='./data')
    texts = ["The quick brown fox.", "Him go store.", "This is correct.", "Me likes it."]
    labels = [1, 0, 1, 0]
    enc = tokenizer(texts, padding=True, truncation=True, max_length=64, return_tensors='pt')
    enc['labels'] = torch.tensor(labels)
    batch = {k: v.to(device) for k, v in enc.items()}

    targets = ['query', 'value'] if 'bert' in model_name else ['q_proj', 'v_proj']

    for method, scaling in [('spectral_s1', 1.0), ('spectral_s20', 20.0),
                            ('spectral_s100', 100.0), ('spectral_s150', 150.0)]:
        model = AutoModelForSequenceClassification.from_pretrained(
            model_name, num_labels=2, cache_dir='./data'
        )
        model = get_spectral_adapter_model(
            model, target_modules=targets,
            p=16, q=16, scaling=scaling, d_initial=0.01,
            freq_mode='contiguous'
        )
        model = model.to(device)
        model.train()
        outputs = model(**batch)
        outputs.loss.backward()

        adapter_grads = []
        classifier_grads = []
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                g = param.grad.norm().item() / math.sqrt(param.numel())
                if 'coeffs' in name:
                    adapter_grads.append(g)
                elif 'classifier' in name or 'score' in name:
                    classifier_grads.append(g)

        avg_adapter_grad = np.mean(adapter_grads) if adapter_grads else 0
        avg_classifier_grad = np.mean(classifier_grads) if classifier_grads else 0
        ratio = avg_adapter_grad / (avg_classifier_grad + 1e-12)

        print(f"  {method:20s}: adapter_grad_rms={avg_adapter_grad:.6e}, classifier_grad_rms={avg_classifier_grad:.6e}, ratio={ratio:.4f}, loss={outputs.loss.item():.4f}")
        del model

    # FourierFT
    model = AutoModelForSequenceClassification.from_pretrained(
        model_name, num_labels=2, cache_dir='./data'
    )
    config = FourierFTConfig(target_modules=targets, n_frequency=256, scaling=150.0, random_loc_seed=42)
    model = get_peft_model(model, config).to(device)
    model.train()
    outputs = model(**batch)
    outputs.loss.backward()

    adapter_grads = []
    classifier_grads = []
    for name, param in model.named_parameters():
        if param.requires_grad and param.grad is not None:
            g = param.grad.norm().item() / math.sqrt(param.numel())
            if 'spectrum' in name:
                adapter_grads.append(g)
            elif 'classifier' in name or 'score' in name:
                classifier_grads.append(g)

    avg_adapter_grad = np.mean(adapter_grads) if adapter_grads else 0
    avg_classifier_grad = np.mean(classifier_grads) if classifier_grads else 0
    ratio = avg_adapter_grad / (avg_classifier_grad + 1e-12)
    print(f"  {'fourierft_s150':20s}: adapter_grad_rms={avg_adapter_grad:.6e}, classifier_grad_rms={avg_classifier_grad:.6e}, ratio={ratio:.4f}, loss={outputs.loss.item():.4f}")
    del model


if __name__ == '__main__':
    print("ROOT CAUSE ANALYSIS: Spectral failure on RoBERTa CoLA")
    print("=" * 70)

    # Part 1: Compare adapter output magnitudes
    print("\n### PART 1: Adapter output magnitudes ###")
    measure_adapter_magnitudes('roberta-base', device='cpu')
    measure_adapter_magnitudes('bert-base-uncased', device='cpu')

    # Part 2: Gradient analysis
    print("\n### PART 2: Gradient analysis ###")
    gradient_analysis('roberta-base', device='cpu')
    gradient_analysis('bert-base-uncased', device='cpu')

    print("\n\nDONE.")
