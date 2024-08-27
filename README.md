# Speech2Speech Agent Implementation

**Video Demo:**
[Link to the Video](https://drive.google.com/file/d/1ymxYmA2lvG2fvs3zlgKa4WgLlf9dtXSA/view?usp=sharing)

### Solution Overview:

The solution implements a Speech-to-Speech pipeline, converting an audio input into synthesized speech output using a series of models and libraries. The pipeline processes audio input by first detecting voice activity, transcribing speech to text, generating a text-based response using a language model, and finally converting the text back into speech.

### Key Components and Choices:

1. **Voice Activity Detection (VAD):**
   - **Model Used:** `silero-vad`
   - **Purpose:** This model detects segments of speech within an audio file, isolating them from silence or noise.
   - **Libraries:** `torch.hub`, `snakers4/silero-vad` repo
   - **Parameters:** ONNX model for faster inference, 16 kHz sampling rate for audio processing.

2. **Speech-to-Text (Whisper):**
   - **Model Used:** `openai/whisper-small` (Processor) and `openai/whisper-tiny` (Model)
   - **Purpose:** Converts detected speech segments into text.
   - **Libraries:** `WhisperProcessor`, `WhisperForConditionalGeneration` from Hugging Face
   - **Parameters:**
     - `device_map="auto"`: Automatically assigns the model to the available GPU.
     - `load_in_8bit=True`: Loads the model in 8-bit precision to reduce memory usage.

3. **Response Generation (LLaMA):**
   - **Model Used:** `microsoft/Phi-3.5-mini-instruct`
   - **Purpose:** Generates a concise and precise text response based on the transcribed text input.
   - **Libraries:** `AutoTokenizer`, `AutoModelForCausalLM` from Hugging Face
   - **Parameters:**
     - `load_in_4bit=True`: Loads the model in 4-bit precision to optimize GPU memory.
     - `device_map="auto"`: For GPU assignment.
     - `max_memory={0: "15GB"}`: Allocates 15GB of GPU memory.
     - `max_new_tokens=64`: Limits the response generation to 64 tokens.
     - `temperature=0.1`: Low temperature for more deterministic outputs.
     - `top_p=0.9`: Controls the diversity of the generated responses.

4. **Text-to-Speech (Parler TTS):**
   - **Model Used:** `parler-tts/parler-tts-mini-expresso`
   - **Purpose:** Converts the generated text response back into speech.
   - **Libraries:** `ParlerTTSForConditionalGeneration`, `AutoTokenizer` from Hugging Face
   - **Parameters:**
     - Custom description for voice modulation.
     - CUDA-based processing for faster inference.
     - `soundfile` library to save the synthesized speech.

### Pipeline Execution:
1. **Audio Processing:** The input audio file is processed to extract speech segments using VAD.
2. **Speech-to-Text:** The speech segments are transcribed into text using the Whisper model.
3. **Response Generation:** The transcribed text is fed into the LLaMA model, which generates a concise response.
4. **Text-to-Speech:** The generated text response is synthesized into speech using the Parler TTS model and saved as an output file.

### Summary:
This solution leverages state-of-the-art models for each stage of the pipeline, optimizing performance with quantization techniques (8-bit and 4-bit) and GPU memory management. It provides a streamlined process for real-time or near-real-time speech-to-speech conversion, suitable for applications like virtual assistants or voice-interactive systems.

**Current Status:**
I have made significant progress in implementing a Speech2Speech agent with the goal of creating a real-time demo using a Streamlit app. However, the integration with ngrok encountered issues, preventing full deployment. WebRTC could be a viable alternative for streaming audio from a browser microphone to a server, but due to time constraints from mid-term examinations, I was unable to implement it. I plan to continue developing this project to achieve near real-time performance.

**Performance Enhancement Suggestions:**
The current solution has a latency of 21 seconds. This can be reduced by streaming the audio output using one of the following methods:
- **XTTS V2:** Tokenizes input text and feeds it to a speech synthesis model, producing audio chunks.
- **Orca Streaming Text-to-Speech:** Synthesizes audio while the Large Language Model (LLM) is still generating a response.

Additionally, utilizing a more recent GPU architecture that supports FP4 or faster INT4 kernels could significantly reduce output times, enhancing the real-time capabilities of the system.

**LLM Routing:**
Incorporating LLM routing can optimize performance by assessing the complexity of the query and directing it to an appropriately sized LLM. This can be implemented using [RouteLLM](https://github.com/lm-sys/RouteLLM), which would improve both cost efficiency and response time.

**References:**
- [LiveKit Voice Assistant](https://github.com/livekit/agents/tree/main/examples/voice-assistant)
- [HuggingFace Speech-to-Speech](https://github.com/huggingface/speech-to-speech)
