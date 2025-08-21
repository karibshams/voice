def start_recording():
    """Start recording audio from microphone."""
    try:
        import sounddevice as sd
        
        st.session_state.is_recording = True
        st.session_state.audio_data = []
        
        # Recording parameters
        sample_rate = 44100
        channels = 1
        
        def audio_callback(indata, frames, time, status):
            """Callback for audio input."""
            if st.session_state.is_recording:
                st.session_state.audio_data.extend(indata[:, 0])
        
        # Start recording in a separate thread
        def record_audio():
            with sd.InputStream(callback=audio_callback, 
                              samplerate=sample_rate, 
                              channels=channels):
                while st.session_state.is_recording:
                    time.sleep(0.1)
        
        # Start recording thread
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.daemon = True
        recording_thread.start()
        
        st.rerun()  # Refresh to show recording status
        
    except Exception as e:
        st.error(f"❌ Failed to start recording: {str(e)}")
        st.session_state.is_recording = False

def stop_recording():
    """Stop recording and process audio."""
    if not st.session_state.is_recording:
        return
    
    st.session_state.is_recording = False
    
    if not st.session_state.audio_data:
        st.warning("⚠️ No audio data recorded")
        return
    
    # Check recording duration
    duration = len(st.session_state.audio_data) / 44100
    if duration < 1.0:
        st.warning(f"⚠️ Recording too short ({duration:.1f}s). Try speaking for at least 1 second.")
        return
    
    st.info(f"📊 Processing {duration:.1f} seconds of audio...")
    
    # Convert audio to bytes for Whisper
    try:
        audio_array = np.array(st.session_state.audio_data, dtype=np.float32)
        audio_bytes = convert_audio_to_wav_bytes(audio_array)
        
        # Use the chatbot's speech-to-text function
        with st.spinner("🔄 Converting speech to text..."):
            transcription = st.session_state.chatbot_instance.speech_to_text(audio_bytes)
            
        if transcription:
            st.session_state.last_transcription = transcription
            st.success(f"✅ Transcription successful!")
        else:
            st.error("❌ Failed to transcribe audio")
            
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
    
    st.rerun()  # Refresh to show results

def convert_audio_to_wav_bytes(audio_data):
    """Convert numpy audio data to WAV bytes."""
    # Convert float32 to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes for int16
        wav_file.setframerate(44100)  # Sample rate
        wav_file.writeframes(audio_int16.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()
def run_interactive_test():
    """Run an interactive test with the chatbot."""
    if not st.session_state.chatbot_instance:
        st.error("❌ Chatbot not initialized. Run the full test suite first.")
        return
    
    st.subheader("🤖 Interactive Chat Test")
    st.write("Test the chat functionality with text input (without voice)")
    
    # Text input for testing
    test_message = st.text_input(
        "Enter a test message:",
        value="Hello, this is a test message.",
        help="This tests the chat functionality without voice input"
    )
    
    if st.button("Send Test Message", type="primary"):
        if test_message.strip():
            with st.spinner("Getting chatbot response..."):
                try:
                    response = st.session_state.chatbot_instance.get_chat_response(test_message)
                    
                    if response:
                        st.success("✅ Chat response received!")
                        st.info(f"**You:** {test_message}")
                        st.info(f"**Bot:** {response}")
                    else:
                        st.error("❌ No response received from chatbot")
                        
                except Exception as e:
                    st.error(f"❌ Interactive test failed: {str(e)}")
        else:
            st.warning("Please enter a test message")
import os
import sys
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import threading
from datetime import datetime
import io
import wave
import asyncio

# Load environment variables
load_dotenv()

# Streamlit page config
st.set_page_config(
    page_title="Voice Chatbot Test Suite",
    page_icon="🎤",
    layout="wide",
    initial_sidebar_state="expanded"
)

def init_session_state():
    """Initialize session state variables."""
    if 'test_results' not in st.session_state:
        st.session_state.test_results = {}
    if 'chatbot_instance' not in st.session_state:
        st.session_state.chatbot_instance = None
    if 'test_completed' not in st.session_state:
        st.session_state.test_completed = False
    if 'is_recording' not in st.session_state:
        st.session_state.is_recording = False
    if 'audio_data' not in st.session_state:
        st.session_state.audio_data = []
    if 'last_transcription' not in st.session_state:
        st.session_state.last_transcription = ""
    if 'voice_test_results' not in st.session_state:
        st.session_state.voice_test_results = []
    if 'recording_start_time' not in st.session_state:
        st.session_state.recording_start_time = 0
    if 'recording_duration' not in st.session_state:
        st.session_state.recording_duration = 0
    if 'recorded_audio' not in st.session_state:
        st.session_state.recorded_audio = None
    if 'selected_device' not in st.session_state:
        st.session_state.selected_device = None
    if 'recording_progress' not in st.session_state:
        st.session_state.recording_progress = 0.0
    if 'current_audio_level' not in st.session_state:
        st.session_state.current_audio_level = 0.0

def test_environment():
    """Test if environment variables are properly set."""
    with st.spinner("Testing environment setup..."):
        results = {
            'status': 'success',
            'details': [],
            'warnings': []
        }
        
        api_key = os.getenv('OPENAI_API_KEY')
        stt_model = os.getenv('STT_MODEL', 'whisper-1')
        chat_model = os.getenv('CHAT_MODEL', 'gpt-4o-mini')
        
        if not api_key:
            results['status'] = 'error'
            results['details'].append("❌ OPENAI_API_KEY not found in .env file")
            return results
        
        if api_key.startswith('sk-'):
            results['details'].append("✅ OpenAI API key format looks correct")
        else:
            results['status'] = 'warning'
            results['warnings'].append("⚠️ API key format might be incorrect (should start with 'sk-')")
        
        results['details'].extend([
            f"✅ STT Model: {stt_model}",
            f"✅ Chat Model: {chat_model}"
        ])
        
        return results

def test_dependencies():
    """Test if all required dependencies are available."""
    with st.spinner("Testing dependencies..."):
        results = {
            'status': 'success',
            'details': [],
            'missing': []
        }
        
        required_modules = [
            ('openai', 'OpenAI'),
            ('sounddevice', 'sounddevice'),
            ('numpy', 'numpy'),
            ('wave', 'wave'),
            ('io', 'io'),
            ('threading', 'threading'),
            ('streamlit', 'streamlit')
        ]
        
        for module_name, import_name in required_modules:
            try:
                __import__(import_name if import_name else module_name)
                results['details'].append(f"✅ {module_name}")
            except ImportError:
                results['status'] = 'error'
                results['details'].append(f"❌ {module_name} - NOT FOUND")
                results['missing'].append(module_name)
        
        if results['missing']:
            results['details'].append(f"\n💡 Install missing: pip install {' '.join(results['missing'])}")
        
        return results

def test_audio_devices():
    """Test audio input devices."""
    with st.spinner("Testing audio devices..."):
        results = {
            'status': 'success',
            'details': [],
            'devices': []
        }
        
        try:
            import sounddevice as sd
            
            # Get available devices
            devices = sd.query_devices()
            input_devices = [d for d in devices if d['max_input_channels'] > 0]
            
            if not input_devices:
                results['status'] = 'error'
                results['details'].append("❌ No audio input devices found")
                return results
            
            results['details'].append(f"✅ Found {len(input_devices)} input device(s):")
            for i, device in enumerate(input_devices):
                device_info = f"   {i}: {device['name']}"
                results['details'].append(device_info)
                results['devices'].append(device['name'])
            
            # Test recording capability
            try:
                duration = 0.5  # Short test
                sample_rate = 44100
                test_recording = sd.rec(int(duration * sample_rate), 
                                      samplerate=sample_rate, channels=1)
                sd.wait()
                results['details'].append("✅ Microphone access working")
                
            except Exception as e:
                results['status'] = 'warning'
                results['details'].append(f"⚠️ Microphone test warning: {str(e)}")
                
        except ImportError:
            results['status'] = 'error'
            results['details'].append("❌ sounddevice not available")
        
        return results

def test_openai_connection():
    """Test OpenAI API connection."""
    with st.spinner("Testing OpenAI API connection..."):
        results = {
            'status': 'success',
            'details': [],
            'response': None
        }
        
        try:
            from openai import OpenAI
            
            client = OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
            
            # Test with a simple chat completion
            response = client.chat.completions.create(
                model=os.getenv('CHAT_MODEL', 'gpt-4o-mini'),
                messages=[{"role": "user", "content": "Say 'Hello, test successful!'"}],
                max_tokens=10
            )
            
            result = response.choices[0].message.content
            results['details'].extend([
                "✅ Chat API connection successful",
                f"💬 Test response: {result}"
            ])
            results['response'] = result
            
            results['details'].append("ℹ️ Whisper STT will be tested during actual voice input")
            
        except Exception as e:
            results['status'] = 'error'
            results['details'].append(f"❌ OpenAI API test failed: {str(e)}")
        
        return results

def test_voice_chatbot_import():
    """Test if voice.py can be imported successfully."""
    with st.spinner("Testing voice chatbot import..."):
        results = {
            'status': 'success',
            'details': [],
            'chatbot': None
        }
        
        try:
            sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
            from voice import VoiceChatbot
            
            chatbot = VoiceChatbot()
            results['details'].append("✅ VoiceChatbot created successfully")
            results['chatbot'] = chatbot
            
            # Test components
            if hasattr(chatbot, 'audio_buffer') and isinstance(chatbot.audio_buffer, list):
                results['details'].append("✅ Audio buffer initialized")
            else:
                results['status'] = 'warning'
                results['details'].append("⚠️ Audio buffer not properly initialized")
            
            if hasattr(chatbot, 'client'):
                results['details'].append("✅ OpenAI client initialized")
            else:
                results['status'] = 'warning'
                results['details'].append("⚠️ OpenAI client not found")
            
            # Store chatbot instance in session state
            st.session_state.chatbot_instance = chatbot
            
        except Exception as e:
            results['status'] = 'error'
            results['details'].append(f"❌ Voice chatbot import failed: {str(e)}")
        
        return results

def run_interactive_test():
    """Run an interactive test with the chatbot."""
    if not st.session_state.chatbot_instance:
        st.error("❌ Chatbot not initialized. Run the full test suite first.")
        return
    
    st.subheader("🤖 Interactive Chat Test")
    st.write("Test the chat functionality with text input (without voice)")
    
    # Text input for testing
    test_message = st.text_input(
        "Enter a test message:",
        value="Hello, this is a test message.",
        help="This tests the chat functionality without voice input"
    )
    
    if st.button("Send Test Message", type="primary"):
        if test_message.strip():
            with st.spinner("Getting chatbot response..."):
                try:
                    response = st.session_state.chatbot_instance.get_chat_response(test_message)
                    
                    if response:
                        st.success("✅ Chat response received!")
                        st.info(f"**You:** {test_message}")
                        st.info(f"**Bot:** {response}")
                    else:
                        st.error("❌ No response received from chatbot")
                        
                except Exception as e:
                    st.error(f"❌ Interactive test failed: {str(e)}")
        else:
            st.warning("Please enter a test message")

def run_voice_test():
    """Run real-time voice recording and transcription test."""
    if not st.session_state.chatbot_instance:
        st.error("❌ Chatbot not initialized. Run the full test suite first.")
        return
    
    st.subheader("🎤 Real-Time Voice Test")
    st.write("Test your microphone and speech-to-text functionality in real-time!")
    
    # Important note about browser limitations
    st.info("ℹ️ **Note:** Browser-based audio recording has limitations. For full testing, use `python voice.py` in terminal.")
    
    # Alternative testing approaches
    st.markdown("### 🔄 Voice Testing Options:")
    
    # Option 1: File upload test
    st.markdown("#### Option 1: Upload Audio File")
    uploaded_file = st.file_uploader(
        "Upload an audio file (WAV, MP3, M4A) to test transcription:",
        type=['wav', 'mp3', 'm4a', 'ogg'],
        help="Record audio on your phone/computer and upload it here to test Whisper transcription"
    )
    
    if uploaded_file is not None:
        st.success(f"✅ Audio file uploaded: {uploaded_file.name}")
        
        if st.button("🔄 Transcribe Uploaded Audio", type="primary"):
            with st.spinner("🔄 Converting speech to text..."):
                try:
                    # Read the uploaded file
                    audio_bytes = uploaded_file.read()
                    
                    # Use the chatbot's speech-to-text function
                    transcription = st.session_state.chatbot_instance.speech_to_text(audio_bytes)
                    
                    if transcription:
                        st.success("✅ Transcription successful!")
                        st.write(f"**Transcription:** {transcription}")
                        st.session_state.last_transcription = transcription
                        
                        # Test chat response
                        if st.button("🤖 Get Chat Response", key="file_chat_test"):
                            with st.spinner("Getting chatbot response..."):
                                response = st.session_state.chatbot_instance.get_chat_response(transcription)
                                if response:
                                    st.info(f"**Bot Response:** {response}")
                                    # Store successful test
                                    st.session_state.voice_test_results.append({
                                        'timestamp': datetime.now().strftime('%H:%M:%S'),
                                        'transcription': transcription,
                                        'response': response,
                                        'status': 'success',
                                        'method': 'file_upload'
                                    })
                                else:
                                    st.error("❌ No response from chatbot")
                    else:
                        st.error("❌ Failed to transcribe audio")
                        
                except Exception as e:
                    st.error(f"❌ Error processing audio file: {str(e)}")
    
    st.divider()
    
    # Option 2: Live recording (with improved implementation)
    st.markdown("#### Option 2: Live Recording")
    st.info("🎤 **Live Voice Recording:** Test your microphone in real-time with improved audio processing")
    
    # Audio device selection
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devices = [(i, d['name']) for i, d in enumerate(devices) if d['max_input_channels'] > 0]
        
        if input_devices:
            device_names = [name for _, name in input_devices]
            selected_device = st.selectbox(
                "Select Microphone Device:",
                options=range(len(device_names)),
                format_func=lambda x: device_names[x],
                help="Choose your preferred microphone"
            )
            st.session_state.selected_device = input_devices[selected_device][0]
        else:
            st.error("❌ No input devices found")
            return
            
    except Exception as e:
        st.error(f"❌ Could not detect audio devices: {e}")
        return
    
    # Recording settings
    col_settings1, col_settings2 = st.columns(2)
    with col_settings1:
        duration = st.slider("Recording Duration (seconds)", 1, 10, 3, help="How long to record")
    with col_settings2:
        sample_rate = st.selectbox("Sample Rate", [16000, 22050, 44100], index=0, help="Audio quality")
    
    # Recording controls
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        if st.button("🔴 Start Recording", disabled=st.session_state.is_recording, use_container_width=True, type="primary"):
            start_live_recording(duration, sample_rate)
    
    with col2:
        if st.button("⏹️ Stop Recording", disabled=not st.session_state.is_recording, use_container_width=True):
            stop_live_recording()
    
    with col3:
        if st.session_state.is_recording:
            st.error("🔴 Recording... Speak now!")
        else:
            st.success("Ready to record")
    
    # Real-time recording feedback
    if st.session_state.is_recording:
        progress_container = st.container()
        audio_level_container = st.container()
        
        # Progress bar for recording duration
        if hasattr(st.session_state, 'recording_progress'):
            with progress_container:
                progress = min(st.session_state.recording_progress, 1.0)
                st.progress(progress, text=f"Recording: {progress*100:.0f}%")
        
        # Audio level indicator
        if hasattr(st.session_state, 'current_audio_level'):
            with audio_level_container:
                level = st.session_state.current_audio_level
                st.metric("🎤 Audio Level", f"{level:.1f}%", 
                         delta=f"{'Good' if level > 10 else 'Speak louder'}")
    
    # Show recording results
    if hasattr(st.session_state, 'recorded_audio') and st.session_state.recorded_audio is not None:
        st.success("✅ Recording completed!")
        
        # Audio info
        audio_data = st.session_state.recorded_audio
        duration_actual = len(audio_data) / sample_rate
        max_amplitude = np.max(np.abs(audio_data))
        
        col_info1, col_info2, col_info3 = st.columns(3)
        with col_info1:
            st.metric("Duration", f"{duration_actual:.1f}s")
        with col_info2:
            st.metric("Max Volume", f"{max_amplitude:.3f}")
        with col_info3:
            quality = "Good" if max_amplitude > 0.1 else "Low" if max_amplitude > 0.01 else "Very Low"
            st.metric("Quality", quality)
        
        # Transcribe button
        if st.button("🔄 Transcribe Audio", type="primary", use_container_width=True):
            transcribe_recorded_audio(audio_data, sample_rate)
    
    st.divider()
    
    # Option 3: Terminal recommendation
    st.markdown("#### Option 3: Full Terminal Testing (Recommended)")
    st.info("""
    **🚀 For best results, test the complete voice functionality in terminal:**
    
    ```bash
    python voice.py
    ```
    
    This provides:
    - ✅ Real-time microphone access
    - ✅ Continuous recording
    - ✅ Better audio quality
    - ✅ No browser limitations
    """)
    
    # Previous test results
    if st.session_state.voice_test_results:
        st.divider()
        st.subheader("📋 Previous Voice Tests")
        for i, test in enumerate(reversed(st.session_state.voice_test_results[-5:])):
            method_emoji = "📁" if test.get('method') == 'file_upload' else "🎤"
            with st.expander(f"{method_emoji} Test {len(st.session_state.voice_test_results) - i} - {test['timestamp']}", expanded=i==0):
                st.write(f"**Method:** {test.get('method', 'unknown')}")
                st.write(f"**You said:** {test['transcription']}")
                st.write(f"**Bot replied:** {test['response']}")
                st.success("✅ Voice-to-text-to-response pipeline working!")

def start_live_recording(duration, sample_rate):
    """Start live recording with improved real-time feedback."""
    try:
        import sounddevice as sd
        
        st.session_state.is_recording = True
        st.session_state.recording_start_time = time.time()
        st.session_state.recording_progress = 0.0
        st.session_state.current_audio_level = 0.0
        
        # Recording parameters
        channels = 1
        device_id = getattr(st.session_state, 'selected_device', None)
        
        # Pre-allocate audio buffer
        total_frames = int(duration * sample_rate)
        st.session_state.recorded_audio = np.zeros(total_frames, dtype=np.float32)
        current_frame = 0
        
        def audio_callback(indata, frames, time_info, status):
            """Real-time audio callback with feedback."""
            nonlocal current_frame
            
            if not st.session_state.is_recording:
                return
            
            # Calculate audio level for feedback
            audio_level = np.sqrt(np.mean(indata**2)) * 100
            st.session_state.current_audio_level = audio_level
            
            # Store audio data
            if current_frame + frames <= total_frames:
                st.session_state.recorded_audio[current_frame:current_frame + frames] = indata[:, 0]
                current_frame += frames
                
                # Update progress
                st.session_state.recording_progress = current_frame / total_frames
                
            if current_frame >= total_frames:
                st.session_state.is_recording = False
        
        def record_with_callback():
            """Record audio with callback in separate thread."""
            try:
                with sd.InputStream(
                    callback=audio_callback,
                    device=device_id,
                    samplerate=sample_rate,
                    channels=channels,
                    dtype=np.float32
                ):
                    # Record for specified duration
                    start_time = time.time()
                    while st.session_state.is_recording and (time.time() - start_time) < duration:
                        time.sleep(0.1)
                        
                # Ensure recording stops
                st.session_state.is_recording = False
                
            except Exception as e:
                st.session_state.is_recording = False
                st.error(f"Recording error: {str(e)}")
        
        # Start recording thread
        recording_thread = threading.Thread(target=record_with_callback)
        recording_thread.daemon = True
        recording_thread.start()
        
        # Auto-refresh for real-time feedback
        time.sleep(0.1)
        st.rerun()
        
    except Exception as e:
        st.error(f"❌ Failed to start recording: {str(e)}")
        st.session_state.is_recording = False

def stop_live_recording():
    """Stop live recording."""
    if st.session_state.is_recording:
        st.session_state.is_recording = False
        st.success("⏹️ Recording stopped by user")
        time.sleep(0.2)  # Allow callback to finish
        st.rerun()

def transcribe_recorded_audio(audio_data, sample_rate):
    """Transcribe the recorded audio."""
    try:
        with st.spinner("🔄 Converting speech to text..."):
            # Process and normalize audio
            if np.max(np.abs(audio_data)) > 0:
                # Normalize audio to prevent clipping
                audio_data = audio_data / np.max(np.abs(audio_data)) * 0.8
            
            # Convert to WAV bytes
            audio_bytes = convert_audio_to_wav_bytes_improved(audio_data, sample_rate)
            
            # Use chatbot's speech-to-text
            transcription = st.session_state.chatbot_instance.speech_to_text(audio_bytes)
            
            if transcription and transcription.strip():
                st.session_state.last_transcription = transcription.strip()
                st.success("✅ Transcription successful!")
                st.write(f"**You said:** {transcription}")
                
                # Offer chat response
                if st.button("🤖 Get Chat Response", key="transcribed_chat", type="primary"):
                    with st.spinner("Getting bot response..."):
                        response = st.session_state.chatbot_instance.get_chat_response(transcription)
                        if response:
                            st.info(f"**Bot replied:** {response}")
                            
                            # Store test result
                            st.session_state.voice_test_results.append({
                                'timestamp': datetime.now().strftime('%H:%M:%S'),
                                'transcription': transcription,
                                'response': response,
                                'status': 'success',
                                'method': 'live_recording_improved'
                            })
                        else:
                            st.error("❌ No response from bot")
                            
            else:
                st.error("❌ No speech detected in audio")
                st.write("💡 **Tips for better results:**")
                st.write("- Speak clearly and loudly")
                st.write("- Reduce background noise")
                st.write("- Get closer to microphone")
                
    except Exception as e:
        st.error(f"❌ Transcription failed: {str(e)}")

def start_recording_improved():
    """Improved recording start with better error handling."""
    try:
        import sounddevice as sd
        
        st.session_state.is_recording = True
        st.session_state.audio_data = []
        st.session_state.recording_start_time = time.time()
        
        # Show immediate feedback
        st.success("🔴 Recording started! Speak now...")
        
        # Simple recording approach
        sample_rate = 16000  # Lower sample rate for better compatibility
        channels = 1
        
        def record_chunk():
            """Record a small chunk of audio."""
            try:
                chunk_duration = 0.1  # 100ms chunks
                frames = int(sample_rate * chunk_duration)
                
                while st.session_state.is_recording:
                    chunk = sd.rec(frames, samplerate=sample_rate, channels=channels, dtype=np.float32)
                    sd.wait()
                    
                    if st.session_state.is_recording:  # Check again in case stopped during recording
                        st.session_state.audio_data.extend(chunk[:, 0])
                        st.session_state.recording_duration = time.time() - st.session_state.recording_start_time
                    
                    time.sleep(0.05)  # Small delay
                    
            except Exception as e:
                st.error(f"Recording error: {str(e)}")
                st.session_state.is_recording = False
        
        # Start recording in background thread
        recording_thread = threading.Thread(target=record_chunk)
        recording_thread.daemon = True
        recording_thread.start()
        
        st.rerun()
        
    except ImportError:
        st.error("❌ sounddevice not available. Please install it with: pip install sounddevice")
        st.session_state.is_recording = False
    except Exception as e:
        st.error(f"❌ Failed to start recording: {str(e)}")
        st.session_state.is_recording = False

def stop_recording_improved():
    """Improved recording stop with better processing."""
    if not st.session_state.is_recording:
        return
    
    st.session_state.is_recording = False
    time.sleep(0.2)  # Give time for last chunks to be recorded
    
    if not st.session_state.audio_data:
        st.warning("⚠️ No audio data recorded. Try speaking louder or check microphone permissions.")
        return
    
    duration = len(st.session_state.audio_data) / 16000
    if duration < 0.5:
        st.warning(f"⚠️ Recording too short ({duration:.1f}s). Try speaking for at least 1 second.")
        return
    
    st.info(f"📊 Processing {duration:.1f} seconds of audio...")
    
    try:
        # Convert to proper format
        audio_array = np.array(st.session_state.audio_data, dtype=np.float32)
        
        # Normalize audio
        if np.max(np.abs(audio_array)) > 0:
            audio_array = audio_array / np.max(np.abs(audio_array)) * 0.8
        
        # Convert to WAV bytes
        audio_bytes = convert_audio_to_wav_bytes_improved(audio_array, 16000)
        
        with st.spinner("🔄 Converting speech to text..."):
            transcription = st.session_state.chatbot_instance.speech_to_text(audio_bytes)
            
        if transcription and transcription.strip():
            st.session_state.last_transcription = transcription.strip()
            st.success(f"✅ Transcription successful!")
        else:
            st.error("❌ No speech detected. Try speaking more clearly or closer to the microphone.")
            
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
        st.write("💡 **Tip:** Try using the file upload option or run `python voice.py` in terminal for better results.")
    
    st.rerun()

def convert_audio_to_wav_bytes_improved(audio_data, sample_rate):
    """Improved audio conversion with proper formatting."""
    # Ensure audio is in the right range
    audio_data = np.clip(audio_data, -1.0, 1.0)
    
    # Convert to 16-bit integers
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes per sample (16-bit)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_int16.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()

def start_recording():
    """Start recording audio from microphone."""
    try:
        import sounddevice as sd
        
        st.session_state.is_recording = True
        st.session_state.audio_data = []
        
        # Recording parameters
        sample_rate = 44100
        channels = 1
        
        def audio_callback(indata, frames, time, status):
            """Callback for audio input."""
            if st.session_state.is_recording:
                st.session_state.audio_data.extend(indata[:, 0])
        
        # Start recording in a separate thread
        def record_audio():
            with sd.InputStream(callback=audio_callback, 
                              samplerate=sample_rate, 
                              channels=channels):
                while st.session_state.is_recording:
                    time.sleep(0.1)
        
        # Start recording thread
        recording_thread = threading.Thread(target=record_audio)
        recording_thread.daemon = True
        recording_thread.start()
        
        st.rerun()  # Refresh to show recording status
        
    except Exception as e:
        st.error(f"❌ Failed to start recording: {str(e)}")
        st.session_state.is_recording = False

def stop_recording():
    """Stop recording and process audio."""
    if not st.session_state.is_recording:
        return
    
    st.session_state.is_recording = False
    
    if not st.session_state.audio_data:
        st.warning("⚠️ No audio data recorded")
        return
    
    # Check recording duration
    duration = len(st.session_state.audio_data) / 44100
    if duration < 1.0:
        st.warning(f"⚠️ Recording too short ({duration:.1f}s). Try speaking for at least 1 second.")
        return
    
    st.info(f"📊 Processing {duration:.1f} seconds of audio...")
    
    # Convert audio to bytes for Whisper
    try:
        audio_array = np.array(st.session_state.audio_data, dtype=np.float32)
        audio_bytes = convert_audio_to_wav_bytes(audio_array)
        
        # Use the chatbot's speech-to-text function
        with st.spinner("🔄 Converting speech to text..."):
            transcription = st.session_state.chatbot_instance.speech_to_text(audio_bytes)
            
        if transcription:
            st.session_state.last_transcription = transcription
            st.success(f"✅ Transcription successful!")
        else:
            st.error("❌ Failed to transcribe audio")
            
    except Exception as e:
        st.error(f"❌ Error processing audio: {str(e)}")
    
    st.rerun()  # Refresh to show results

def convert_audio_to_wav_bytes(audio_data):
    """Convert numpy audio data to WAV bytes."""
    # Convert float32 to int16
    audio_int16 = (audio_data * 32767).astype(np.int16)
    
    # Create WAV file in memory
    wav_buffer = io.BytesIO()
    with wave.open(wav_buffer, 'wb') as wav_file:
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 2 bytes for int16
        wav_file.setframerate(44100)  # Sample rate
        wav_file.writeframes(audio_int16.tobytes())
    
    wav_buffer.seek(0)
    return wav_buffer.read()

def display_test_result(test_name, result):
    """Display test results with appropriate styling."""
    if result['status'] == 'success':
        st.success(f"✅ {test_name} - PASSED")
    elif result['status'] == 'warning':
        st.warning(f"⚠️ {test_name} - PASSED WITH WARNINGS")
    else:
        st.error(f"❌ {test_name} - FAILED")
    
    # Show details
    for detail in result['details']:
        st.write(detail)
    
    if 'warnings' in result and result['warnings']:
        for warning in result['warnings']:
            st.warning(warning)

def main():
    """Main Streamlit application."""
    init_session_state()
    
    # Header
    st.title("🎤 Voice Chatbot Test Suite")
    st.markdown("Comprehensive testing dashboard for your real-time voice chatbot system")
    
    # Sidebar
    with st.sidebar:
        st.header("🔧 Test Controls")
        
        if st.button("Run All Tests", type="primary", use_container_width=True):
            st.session_state.test_completed = False
            st.session_state.test_results = {}
        
        # Clear voice test results
        if st.button("🗑️ Clear Voice Tests", use_container_width=True):
            st.session_state.voice_test_results = []
            st.session_state.last_transcription = ""
            st.success("Voice test history cleared!")
        
        st.divider()
        
        st.header("📋 Individual Tests")
        run_env = st.button("Environment Setup", use_container_width=True)
        run_deps = st.button("Dependencies", use_container_width=True)
        run_audio = st.button("Audio Devices", use_container_width=True)
        run_openai = st.button("OpenAI Connection", use_container_width=True)
        run_import = st.button("Voice Chatbot Import", use_container_width=True)
        
        st.divider()
        
        st.header("ℹ️ System Info")
        st.write(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
        st.write(f"**Python:** {sys.version.split()[0]}")
        st.write(f"**Platform:** {sys.platform}")
    
    # Main content area
    if not st.session_state.test_completed and not any([run_env, run_deps, run_audio, run_openai, run_import]):
        # Welcome screen
        col1, col2, col3 = st.columns([1, 2, 1])
        
        with col2:
            st.markdown("""
            ### 🚀 Welcome to the Voice Chatbot Test Suite
            
            This dashboard will help you verify that your voice chatbot system is properly configured and ready to use.
            
            **What we'll test:**
            - ✅ Environment variables and API keys
            - ✅ Required Python dependencies
            - ✅ Audio devices and microphone access
            - ✅ OpenAI API connectivity
            - ✅ Voice chatbot functionality
            - 🎤 **Real-time voice recording and transcription**
            
            **Click "Run All Tests" in the sidebar to get started!**
            
            After running tests, you can use the **Voice Test** tab to test your microphone and speech-to-text in real-time!
            """)
            
            st.info("💡 **Tip:** Make sure your microphone is connected and you've set up your `.env` file with your OpenAI API key before running tests.")
            
            # Quick microphone permission check
            st.markdown("### 🎤 Microphone Permissions")
            st.write("**Important:** Your browser will ask for microphone permissions when you start voice testing.")
            st.write("Click 'Allow' when prompted to enable voice testing functionality.")
    
    # Run tests based on user input
    tests_to_run = {}
    
    if not st.session_state.test_completed and not any([run_env, run_deps, run_audio, run_openai, run_import]):
        # Run all tests
        if st.session_state.get('test_results') == {}:
            pass  # Don't run tests on initial load
        else:
            tests_to_run = {
                "Environment Setup": test_environment,
                "Dependencies": test_dependencies,
                "Audio Devices": test_audio_devices,
                "OpenAI Connection": test_openai_connection,
                "Voice Chatbot Import": test_voice_chatbot_import
            }
    else:
        # Run individual tests
        if run_env:
            tests_to_run["Environment Setup"] = test_environment
        if run_deps:
            tests_to_run["Dependencies"] = test_dependencies
        if run_audio:
            tests_to_run["Audio Devices"] = test_audio_devices
        if run_openai:
            tests_to_run["OpenAI Connection"] = test_openai_connection
        if run_import:
            tests_to_run["Voice Chatbot Import"] = test_voice_chatbot_import
    
    # Execute tests
    if tests_to_run:
        st.header("🧪 Test Results")
        
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, (test_name, test_func) in enumerate(tests_to_run.items()):
            status_text.text(f"Running {test_name}...")
            progress_bar.progress((i + 1) / len(tests_to_run))
            
            try:
                result = test_func()
                st.session_state.test_results[test_name] = result
            except Exception as e:
                st.session_state.test_results[test_name] = {
                    'status': 'error',
                    'details': [f"❌ Test error: {str(e)}"]
                }
        
        status_text.text("Tests completed!")
        st.session_state.test_completed = True
    
    # Display results
    if st.session_state.test_results:
        st.header("📊 Test Results Summary")
        
        # Results summary
        passed = sum(1 for r in st.session_state.test_results.values() if r['status'] == 'success')
        warnings = sum(1 for r in st.session_state.test_results.values() if r['status'] == 'warning')
        failed = sum(1 for r in st.session_state.test_results.values() if r['status'] == 'error')
        total = len(st.session_state.test_results)
        
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Tests", total)
        col2.metric("✅ Passed", passed, delta=None)
        col3.metric("⚠️ Warnings", warnings, delta=None)
        col4.metric("❌ Failed", failed, delta=None)
        
        # Detailed results
        st.subheader("📋 Detailed Results")
        
        for test_name, result in st.session_state.test_results.items():
            with st.expander(f"{test_name} - {result['status'].upper()}", expanded=result['status'] != 'success'):
                display_test_result(test_name, result)
        
        # Overall status
        if failed == 0:
            if warnings == 0:
                st.balloons()
                st.success("🎉 All tests passed! Your voice chatbot is ready to use.")
                st.info("**Next step:** Run `python voice.py` in your terminal to start the voice chatbot!")
            else:
                st.success("✅ Tests passed with warnings. Your voice chatbot should work, but check the warnings above.")
        else:
            st.error("❌ Some tests failed. Please fix the issues before running the voice chatbot.")
            
            st.markdown("""
            **Common fixes:**
            - Install dependencies: `pip install -r requirements.txt`
            - Set your OpenAI API key in the `.env` file
            - Check microphone permissions and connection
            - Ensure you have sufficient OpenAI API credits
            """)
    
    # Interactive test section
    if st.session_state.test_results and st.session_state.chatbot_instance:
        st.divider()
        
        # Create tabs for different tests
        tab1, tab2 = st.tabs(["💬 Text Chat Test", "🎤 Voice Test"])
        
        with tab1:
            run_interactive_test()
        
        with tab2:
            run_voice_test()
    
    # Footer
    st.divider()
    st.markdown("""
    <div style='text-align: center; color: gray;'>
    Voice Chatbot Test Suite | Built with Streamlit | 
    <a href='https://platform.openai.com/docs' target='_blank'>OpenAI API Docs</a>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()