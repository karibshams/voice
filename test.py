import os
import sys
import time
import numpy as np
import streamlit as st
from dotenv import load_dotenv
import threading
from datetime import datetime

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
            
            **Click "Run All Tests" in the sidebar to get started!**
            """)
            
            st.info("💡 **Tip:** Make sure your microphone is connected and you've set up your `.env` file with your OpenAI API key before running tests.")
    
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
        run_interactive_test()
    
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