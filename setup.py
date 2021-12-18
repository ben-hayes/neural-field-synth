from distutils.core import setup

setup(
    name="neural_field_synth",
    version="0.0.1",
    description="A differentiable synthesiser driven by neural fields built for NASH 2021",
    author="Ben Hayes",
    author_email="b.j.hayes@qmul.ac.uk",
    url="https://github.com/ben-hayes/neural-field-synth",
    packages=["neural_field_synth"],
    install_requires=[
        "auraloss==0.2.1",
        "torch@https://download.pytorch.org/whl/cu113/torch-1.10.1%2Bcu113-cp39-cp39-linux_x86_64.whl",
        "numpy==1.21.4",
        "torchaudio@https://download.pytorch.org/whl/cu113/torchaudio-0.10.1%2Bcu113-cp39-cp39-linux_x86_64.whl",
        "pytorch-lightning==1.5.6",
        "torchtyping==0.1.4",
    ],
)
