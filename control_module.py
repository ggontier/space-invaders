import websocket

def send_command():
    uri = "ws://localhost:8765"
    
    print("Control module connecting to Space Invaders...")
    ws = websocket.create_connection(uri)
    print("Connected! Controls:")
    print("'q' or 'left' = LEFT")
    print("'d' or 'right' = RIGHT")
    print("'space' or 'f' = FIRE")
    print("'enter' = ENTER")
    print("'a' = QUIT")
    
    # Simple mapping of keys to commands
    key_to_command = {
        "q": "LEFT",
        "left": "LEFT",
        "d": "RIGHT",
        "right": "RIGHT",
        "space": "FIRE",
        "f": "FIRE",
        "enter": "ENTER",
        "s": "ENTER"
    }
    
    # Main control loop
    while True:
        # Get input from user
        user_input = input("Enter command: ").lower()
        
        print(f"Received: {user_input}")
        # Check for quit command
        if user_input == "a":
            print("Exiting control module")
            break
            
        # Look up the command
        if user_input in key_to_command:
            command = key_to_command[user_input]
            # Send the command
            ws.send(command)
            print(f"Sent: {command}")
        else:
            print(f"Unknown command: {user_input}")
    
    ws.close()


# Run the control module
if __name__ == "__main__":
    send_command() 