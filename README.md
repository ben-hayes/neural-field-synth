# wavespace synthesiser

this is my [NASH 2021](https://signas-qmul.github.io/nash) project, which may or may not end up working 🤷‍♂️

## what is going on?

imagine you have a big folder of audio samples. wouldn't it be cool if you could seamlessly morph between them according to some axes of control you define?

that's the idea here: to have a synthesiser which encodes the timbral characteristics of all your audio.
it works by learning continuous representations of the harmonic and aperiodic components of the audio signals using conditional neural fields.

## what on earth are neural fields?

first, fields. in the sense the term is used in physics, a field is some quantity (a measurement of some physical quantity, for example) that has values across some system of spatiotemporal co-ordinates.
a neural field is simply a neural network that parameterises a field.
you can think of it like a network that takes in co-ordinates and outputs the value you're interested in.
for example, you might pass in spatial (x, y) co-ordinates and get out the pixel values (r, g, b) of an image.
or, you might pass in temporal co-ordinates (t) and get out the amplitude value (a) of an audio signal.

there are a few ways to achieve this with a neural network. the most successful recent methods use periodic nonlinearities or basis functions, such as [SIREN](https://www.vincentsitzmann.com/siren/), [multiplicative filter networks](https://openreview.net/forum?id=OmtmcPkkhT), and [MLPs with Fourier features](https://bmild.github.io/fourfeat/).
here, we use SIRENs, but there is no reason other methods wouldn't work.

here we stretch the definition of a neural field slightly. our co-ordinate systems are not always spatiotemporal, and our measurements are not always physical quantities.
we use two neural fields for the model in this repo:

1. one which learns a continuous space of perfectly looping waveforms (kind of like a bank of wavetables, but without the table and without the bank, hence _wavespace_) as a function of time and phase
2. one which learns a continuous space of zero-phase FIR filter magnitude responses as a function of time and frequency bin

in theory, these continuous representations should allow us to do fun things like sampling at multiple resolutions.

## ok, sure. but what is a conditional neural field?

well, typically you fit a neural field to a single signal. there is some work looking at learning priors over the space of weights of neural fields to allow them to generalise, but it's quite hard to do (i will be releasing a paper soon on some reasons for this).
in our scenario, though, we want to be able to represent _all the sounds in our dataset_ continuously, which a single neural field lacks the capacity to do.

so, we turn to our trusty friend [FiLM conditioning](https://arxiv.org/pdf/1709.07871.pdf) for help.
we use a separate network to generate the FiLM parameters which are inserted between the layers of our neural field, subtly shifting the activations to achieve vastly different results, whilst hopefully allowing the shared weights to learn a convenient general representation.
i'm not the first to consider this approach for audio synthesis: a [recent paper on FiLM conditioned SIRENs for audio synthesis](https://janzuiderveld.github.io/audio-PCINRs/) had promising results.
however, synthesising audio samples directly is hard.
imposing useful priors like the harmonicity of wavetable synthesis, along with a harmonic-plus-noise signal decomposition, allows us to reduce the complexity of the network's task.

## how does it sound?

more soon!

## and what is the model architecture?

also more soon!
