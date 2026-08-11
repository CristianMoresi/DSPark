// Self-contained compile check for DSPark.h.
//
// Includes nothing but the umbrella header and instantiates every public
// template with both float and double. This is what proves the framework
// compiles outside of any host application: a header that only builds because
// some other translation unit happened to include <algorithm> first will fail
// here, and a template that only ever gets exercised at float will fail here
// the moment it is instantiated at double.
//
// Kept deliberately free of assertions - it is a compile gate, not a test.

#include "../DSPark.h"

namespace {

// Force instantiation of every template that takes a sample type.
template <typename T>
void instantiate()
{
    using namespace dspark;

    AudioSpec spec { 48000.0, 512, 2 };

    // -- Core building blocks -------------------------------------------------
    AudioBuffer<T> buf;
    Biquad<T, 1> biquad;
    Convolver<T> conv;
    Dither<T> dither;
    DryWetMixer<T> dw;
    ADSREnvelope<T> envGen;
    FFTReal<T> fftReal(1024);
    FFTComplex<T> fftComplex(1024);
    FIRFilter<T> fir;
    Hilbert<T> hil;
    Hysteresis<T> hyst;
    LadderFilter<T> ladder;
    ModulationRouter<T> modRouter;
    Oscillator<T> osc;
    Oversampling<T> ov(2);
    Phasor<T> ph;
    Resampler<T> rs;
    RingBuffer<T> ring;
    SampleAndHold<T> sh;
    SmoothedValue<T> smv;
    Smoothers::LinearSmoother linSm;
    Smoothers::ExponentialSmoother expSm;
    SpectralProcessor<T> specProc;
    StateVariableFilter<T> svf;
    TruePeakDetector<T> tpd;
    WaveshapeTable<T> ws;
    WavetableOscillator<T> wto;
    ZeroLatencyConvolver<T> zlc;

    // -- Effects --------------------------------------------------------------
    AlgorithmicReverb<T> ar;
    AutoGain<T> ag;
    Chorus<T> ch;
    Clipper<T> clip;
    Compressor<T> cmp;
    Crossfade<T> xfade;
    CrossoverFilter<T> cf;
    DCBlocker<T> dcb;
    DeEsser<T> dee;
    Delay<T> dly;
    DynamicEQ<T> deq;
    Equalizer<T> eq;
    Expander<T> ex;
    FilterEngine<T> filterEngine;
    FrequencyShifter<T> fs;
    Gain<T> g;
    GranularProcessor<T> gran;
    Limiter<T> lim;
    MultibandCompressor<T> mbc;
    NoiseGate<T> ngt;
    NoiseGenerator<T> ng;
    Panner<T> pan;
    Phaser<T> phr;
    PitchShifter<T> pitchShift;
    Reverb<T> rev;
    RingModulator<T> rm;
    Saturation<T> sat;
    SpectralDenoiser<T> denoiser;
    StereoWidth<T> sw;
    TapeMachine<T> tape;
    TransformerModel<T> xfmr;
    TransientDesigner<T> td;
    Tremolo<T> tr;
    TubePreamp<T> tube;
    Vibrato<T> vib;

    // -- Analysis -------------------------------------------------------------
    EnvelopeFollower<T> envFollow;
    Goertzel<T> goe;
    LevelFollower<T> lvl;
    LoudnessMeter<T> ldm;
    LoudnessNormalizer<T> ldn;
    PhaseCorrelation<T> phaseCorr;
    PitchDetector<T> pd;
    PitchFollower<T> pitchFollow;
    SpectrumAnalyzer<T> sa;
    OnsetDetector<T> onset;
    BeatTracker<T> beats;

#ifndef DSPARK_NO_FILE_IO
    // -- File I/O ------------------------------------------------------------
    AudioFileInfo audioFileInfo;
    WavFile wavFile;
    Mp3File mp3File;
    MidiFile midiFile;
    FlacFile flacFile;
#endif

    // -- Music ----------------------------------------------------------------
    ChordDetector<T> chords;
    KeyDetector<T> keys;

    // Touch everything so the optimiser cannot drop it.
    (void)spec; (void)beats;
    (void)buf; (void)biquad; (void)conv; (void)dither; (void)dw; (void)envGen;
    (void)fftReal; (void)fftComplex; (void)fir; (void)hil; (void)hyst;
    (void)ladder; (void)modRouter; (void)osc; (void)ov; (void)ph; (void)rs;
    (void)ring; (void)sh; (void)smv; (void)linSm; (void)expSm; (void)specProc;
    (void)svf; (void)tpd; (void)ws; (void)wto; (void)zlc;
    (void)ar; (void)ag; (void)ch; (void)clip; (void)cmp; (void)xfade; (void)cf;
    (void)dcb; (void)dee; (void)dly; (void)deq; (void)eq; (void)ex;
    (void)filterEngine; (void)fs; (void)g; (void)gran; (void)lim; (void)mbc;
    (void)ngt; (void)ng; (void)pan; (void)phr; (void)pitchShift; (void)rev;
    (void)rm; (void)sat; (void)denoiser; (void)sw; (void)tape; (void)xfmr;
    (void)td; (void)tr; (void)tube; (void)vib;
    (void)envFollow; (void)goe; (void)lvl; (void)ldm; (void)ldn;
    (void)phaseCorr;
    (void)pd; (void)pitchFollow; (void)sa; (void)onset;
    (void)chords; (void)keys;
#ifndef DSPARK_NO_FILE_IO
    (void)audioFileInfo; (void)wavFile; (void)mp3File; (void)midiFile;
    (void)flacFile;
#endif
}

} // namespace

int main()
{
    instantiate<float>();
    instantiate<double>();
    return 0;
}
