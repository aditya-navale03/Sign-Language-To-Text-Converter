import React, { useState, useRef, useEffect } from 'react';
import './App.css';

function App() {
  const [translation, setTranslation] = useState('');
  const videoRef = useRef(null);
  const canvasRef = useRef(null);
  const ws = useRef(null);

  const startCamera = async () => {
    try {
      console.log('Requesting camera access...');
      const stream = await navigator.mediaDevices.getUserMedia({ 
        video: { 
          width: { ideal: 640 },
          height: { ideal: 480 },
          facingMode: 'user',
          frameRate: { ideal: 30, min: 15 }
        } 
      });
      
      if (videoRef.current) {
        console.log('Got camera stream, setting up video element');
        videoRef.current.srcObject = stream;
        
        // Wait for video to be ready
        return new Promise((resolve) => {
          const onLoaded = () => {
            console.log('Video metadata loaded');
            videoRef.current.play().then(() => {
              console.log('Video is playing');
              resolve();
            }).catch(err => {
              console.error('Error playing video:', err);
              resolve(); // Still resolve to continue
            });
          };

          if (videoRef.current.readyState >= 3) { // HAVE_FUTURE_DATA or greater
            onLoaded();
          } else {
            videoRef.current.onloadedmetadata = onLoaded;
          }
        });
      }
    } catch (err) {
      console.error("Error accessing camera:", err);
      throw err;
    }
  };

  const startTranslation = () => {
    ws.current = new WebSocket('ws://localhost:5000/ws');
    
    ws.current.onmessage = async (event) => {
      try {
        const data = JSON.parse(event.data);
        console.log('Received WebSocket message:', data);
        
        if (data.status === 'success') {
          // Always update translation, even if empty (to clear previous translation)
          setTranslation(data.translation || '');
          
          // Update the canvas with the processed frame if available
          if (data.frame && canvasRef.current) {
            const ctx = canvasRef.current.getContext('2d');
            const img = new Image();
            
            try {
              // Convert hex string back to bytes
              const bytes = new Uint8Array(data.frame.match(/.{1,2}/g).map(byte => parseInt(byte, 16)));
              
              // Create blob URL from the image data
              const blob = new Blob([bytes], { type: 'image/jpeg' });
              const url = URL.createObjectURL(blob);
              
              img.onload = () => {
                // Set canvas size to match image
                canvasRef.current.width = img.width;
                canvasRef.current.height = img.height;
                ctx.clearRect(0, 0, canvasRef.current.width, canvasRef.current.height);
                ctx.drawImage(img, 0, 0);
                URL.revokeObjectURL(url);
              };
              
              img.onerror = (err) => {
                console.error('Error loading processed frame:', err);
                URL.revokeObjectURL(url);
              };
              
              img.onabort = (err) => {
                console.error('Image loading aborted:', err);
                URL.revokeObjectURL(url);
              };
              
              img.src = url;
            } catch (err) {
              console.error('Error processing frame data:', err);
            }
          }
        }
      } catch (err) {
        console.error('Error processing WebSocket message:', err);
      }
    };
    
    ws.current.onopen = () => {
      console.log('WebSocket connected');
      let isSending = false;
      
      const sendFrame = async () => {
        if (!isSending && videoRef.current && ws.current.readyState === WebSocket.OPEN) {
          isSending = true;
          
          try {
            const canvas = document.createElement('canvas');
            canvas.width = videoRef.current.videoWidth;
            canvas.height = videoRef.current.videoHeight;
            const ctx = canvas.getContext('2d');
            
            // Draw the current video frame
            ctx.drawImage(videoRef.current, 0, 0, canvas.width, canvas.height);
            
            // Convert to blob and send
            const blob = await new Promise(resolve => 
              canvas.toBlob(blob => resolve(blob), 'image/jpeg', 0.7)
            );
            
            if (ws.current.readyState === WebSocket.OPEN) {
              ws.current.send(blob);
            }
          } catch (err) {
            console.error('Error sending frame:', err);
          } finally {
            isSending = false;
          }
        }
        
        // Schedule next frame
        if (ws.current && ws.current.readyState === WebSocket.OPEN) {
          // Target ~15fps (1000ms/15 ≈ 66ms per frame)
          setTimeout(() => requestAnimationFrame(sendFrame), 66);
        }
      };
      
      // Start sending frames
      sendFrame();
    };
  };

  useEffect(() => {
    let isMounted = true;
    
    const init = async () => {
      try {
        await startCamera();
        if (isMounted) {
          startTranslation();
        }
      } catch (err) {
        console.error('Failed to initialize camera:', err);
      }
    };
    
    init();
    
    return () => {
      isMounted = false;
      
      if (ws.current) {
        ws.current.close();
      }
      
      if (videoRef.current && videoRef.current.srcObject) {
        const tracks = videoRef.current.srcObject.getTracks();
        tracks.forEach(track => {
          track.stop();
          track.enabled = false;
        });
        videoRef.current.srcObject = null;
      }
    };
  }, []);

  return (
    <div className="App">
      <header className="App-header">
        <h1>Sign Language Translator</h1>
      </header>
      
      <div className="translator-container side-by-side">
        <div className="camera-container">
          <div className="camera-feed">
            <div style={{
              position: 'relative',
              width: '100%',
              paddingBottom: '75%', // 4:3 aspect ratio
              backgroundColor: '#000',
              overflow: 'hidden',
              borderRadius: '8px'
            }}>
              <video 
                ref={videoRef} 
                autoPlay 
                playsInline 
                muted
                style={{ 
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  transform: 'scaleX(-1)', // Mirror the video if not used
                  zIndex: 1,
                  backgroundColor: '#000'
                }}
              />
              <canvas 
                ref={canvasRef}
                style={{
                  position: 'absolute',
                  top: 0,
                  left: 0,
                  width: '100%',
                  height: '100%',
                  objectFit: 'cover',
                  transform: 'scaleX(-1)', // Mirror the canvas if not used
                  zIndex: 2,
                  pointerEvents: 'none' // Allow clicks to pass through to video
                }}
              />
            </div>
          </div>
        </div>
        
        <div className="translation-container">
          <h2>Translation:</h2>
          <div className="translation-box">
            {translation || 'Translation will appear here...'}
          </div>
        </div>
      </div>
    </div>
  );
}

export default App;
