const WebSocket = require('ws');

// Create a WebSocket server on port 8765
const wss = new WebSocket.Server({ port: 8765 });

console.log('WebSocket server running on ws://localhost:8765');

// Key code mapping
const KEYCODES = {
  'LEFT': 37,
  'RIGHT': 39,
  'FIRE': 32, // SPACE
  'ENTER': 13
};

// When a client connects
wss.on('connection', (ws) => {
  console.log('Client connected');
  
  // Handle messages from clients (like Python script)
  ws.on('message', (message) => {
    const command = message.toString().trim().toUpperCase();
    console.log(`Received command: ${command}`);
    
    // Check if the command is valid
    if (KEYCODES[command]) {
      const keyCode = KEYCODES[command];
      
      // Send keydown event
      const keydownEvent = JSON.stringify({
        type: 'keydown',
        keyCode: keyCode
      });
      
      // Broadcast keydown to all connected clients
      wss.clients.forEach((client) => {
        if (client.readyState === WebSocket.OPEN) {
          client.send(keydownEvent);
        }
      });
      
      // After a small delay, send the keyup event
      setTimeout(() => {
        const keyupEvent = JSON.stringify({
          type: 'keyup',
          keyCode: keyCode
        });
        
        // Broadcast keyup to all connected clients
        wss.clients.forEach((client) => {
          if (client.readyState === WebSocket.OPEN) {
            client.send(keyupEvent);
          }
        });
      }, 100); // 100ms delay
    } else {
      console.log(`Unknown command: ${command}`);
    }
  });
  
  ws.on('close', () => {
    console.log('Client disconnected');
  });
}); 