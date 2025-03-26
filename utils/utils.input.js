/** 
 * Input Module
 * This module handles all the input events from the player
 * and is part of the state machine. 
 */

var inputHandler = {
  _isPressed: {},
  RIGHT: 39,
  LEFT: 37,
  SPACE: 32,
  ENTER: 13,

  isDown: function isDown (keyCode) {
    return this._isPressed[keyCode];
  },

  onKeydown: function onKeydown (event) {
    console.log('onKeydown', event.keyCode);
    this._isPressed[event.keyCode] = true;
  },

  onKeyup: function onKeyup (event) {
    console.log('onKeyup', event.keyCode);
    delete this._isPressed[event.keyCode];
  },

  init: function init () {
    console.log('Initiating input handler.');
    
    // Connect to WebSocket
    const socket = new WebSocket('ws://localhost:8765');
    
    // Handle WebSocket events
    socket.onopen = function() {
      console.log('WebSocket connection established');
    };
    
    socket.onmessage = function(event) {
      const data = JSON.parse(event.data);
      
      if (data.type === 'keydown') {
        inputHandler.onKeydown({keyCode: data.keyCode});
      } else if (data.type === 'keyup') {
        inputHandler.onKeyup({keyCode: data.keyCode});
      }
    };
    
    socket.onerror = function(error) {
      console.error('WebSocket error:', error);
    };
    
    socket.onclose = function() {
      console.log('WebSocket connection closed');
    };
  }
};

module.exports = inputHandler;