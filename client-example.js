// Example client code for connecting to the RunPod instance
class SpeechToTextClient {
    constructor(runpodUrl) {
        // Replace with your RunPod endpoint URL
        this.wsUrl = `wss://${runpodUrl}/listen`;
        this.socket = null;
        this.mediaRecorder = null;
        this.isRecording = false;
        this.onTranscription = null;
        this.onError = null;
    }

    // Set callback for transcription results
    setTranscriptionCallback(callback) {
        this.onTranscription = callback;
    }

    // Set callback for errors
    setErrorCallback(callback) {
        this.onError = callback;
    }

    // Start recording and transcribing
    async start() {
        if (this.isRecording) return;

        try {
            // Create WebSocket connection
            this.socket = new WebSocket(this.wsUrl);

            this.socket.onopen = async () => {
                console.log('Connected to transcription service');
                
                // Get microphone access
                const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
                this.mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/webm' });

                // Send audio data when available
                this.mediaRecorder.addEventListener('dataavailable', event => {
                    if (event.data.size > 0 && this.socket.readyState === WebSocket.OPEN) {
                        this.socket.send(event.data);
                    }
                });

                // Start recording
                this.mediaRecorder.start(250);
                this.isRecording = true;
            };

            this.socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                
                if (data.type === 'transcription' && this.onTranscription) {
                    this.onTranscription(data.segments);
                } else if (data.type === 'error' && this.onError) {
                    this.onError(data.error);
                }
            };

            this.socket.onerror = (error) => {
                console.error('WebSocket error:', error);
                if (this.onError) {
                    this.onError('Connection error occurred');
                }
            };

            this.socket.onclose = () => {
                console.log('Connection closed');
                this.stop();
            };

        } catch (error) {
            console.error('Error starting recording:', error);
            if (this.onError) {
                this.onError(error.message);
            }
        }
    }

    // Stop recording and close connection
    stop() {
        if (this.mediaRecorder && this.mediaRecorder.state === 'recording') {
            this.mediaRecorder.stop();
            this.mediaRecorder.stream.getTracks().forEach(track => track.stop());
        }
        
        if (this.socket && this.socket.readyState === WebSocket.OPEN) {
            this.socket.close();
        }
        
        this.isRecording = false;
    }
}

// Example usage:
/*
const client = new SpeechToTextClient('your-runpod-endpoint.runpod.net');

client.setTranscriptionCallback((segments) => {
    segments.forEach(segment => {
        console.log(`[${segment.start.toFixed(2)}s -> ${segment.end.toFixed(2)}s] ${segment.text}`);
        segment.words.forEach(word => {
            console.log(`  Word: ${word.word} (${word.start.toFixed(2)}s - ${word.end.toFixed(2)}s)`);
        });
    });
});

client.setErrorCallback((error) => {
    console.error('Error:', error);
});

// Start recording
client.start();

// Stop recording
// client.stop();
*/ 