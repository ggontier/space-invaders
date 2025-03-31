import websockets
import asyncio


async def send_command():
    uri = "ws://localhost:8765"
    
    print("Control module connecting to Space Invaders...")
    async with websockets.connect(uri) as websocket:
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
        
        # Get the current event loop
        loop = asyncio.get_running_loop()

        # Main control loop
        while True:
            # Get input from user asynchronously using run_in_executor
            user_input = await loop.run_in_executor(None, input, "Enter command: ")
            user_input = user_input.lower()
            
            print(f"Received: {user_input}")
            # Check for quit command
            if user_input == "a":
                print("Exiting control module")
                break
                
            # Look up the command
            if user_input in key_to_command:
                command = key_to_command[user_input]
                # Send the command
                await websocket.send(command)
                print(f"Sent: {command}")
            else:
                print(f"Unknown command: {user_input}")


# Run the control module
if __name__ == "__main__":
    asyncio.run(send_command()) 