import time
import asyncio
from mcp.server.fastmcp import FastMCP

# FastMCP server
mcp = FastMCP("fourier_gr2", stateless_http=True, host="0.0.0.0", port=8000)

@mcp.tool()
async def connect_robot() -> str:
    """Connect to Fourier GR2 robot and prepare for control.
    Must be called before any other robot commands.
    """
    print("[MOCK] Connecting to robot...")
    await asyncio.sleep(1.0)
    print("[MOCK] Robot connected successfully.")
    return "Fourier GR2 robot connected and ready."


@mcp.tool()
async def disconnect_robot() -> str:
    """Disconnect from Fourier GR2 robot and return to stand mode."""
    print("[MOCK] Disconnecting robot...")
    await asyncio.sleep(1.0)
    print("[MOCK] Robot disconnected.")
    return "Fourier GR2 robot disconnected."


@mcp.tool()
async def wave_hand(wave_count: int = 5, wave_speed: float = 0.3) -> str:
    """Make the robot wave its right hand.
        控制机器人执行[挥手]动作。
    Args:
        wave_count: Number of wave cycles (default: 5)
        wave_speed: Time for one wave cycle in seconds, smaller is faster (default: 0.3)
    """
    print(f"[MOCK] Waving hand: count={wave_count}, speed={wave_speed}")
    total_time = wave_count * wave_speed
    # Simulate execution time
    await asyncio.sleep(total_time)
    print("[MOCK] Wave finished.")
    return f"Wave completed with {wave_count} cycles."


@mcp.tool()
async def thumbs_up() -> str:
    """Make the robot give a thumbs up gesture with its right hand."""
    print("[MOCK] Doing thumbs up gesture...")
    await asyncio.sleep(2.0)
    print("[MOCK] Thumbs up finished.")
    return "Thumbs up gesture completed."


@mcp.tool()
async def move_arm_to_position(positions: list) -> str:
    """Move robot right arm to specified joint positions.

    Args:
        positions: List of 7 joint positions for right_manipulator
    """
    if len(positions) != 7:
        return "Error: positions must be a list of 7 values"

    print(f"[MOCK] Moving arm to positions: {positions}")
    await asyncio.sleep(1.0)
    return f"Arm moved to position: {positions}"


@mcp.tool()
async def set_hand_gesture(positions: list) -> str:
    """Set robot right hand finger positions.

    Args:
        positions: List of 6 finger joint positions [index, middle, ring, pinky, thumb_rotation, thumb_flex]
    """
    if len(positions) != 6:
        return "Error: positions must be a list of 6 values"

    print(f"[MOCK] Setting hand gesture: {positions}")
    await asyncio.sleep(0.5)
    return f"Hand gesture set to: {positions}"


if __name__ == "__main__":
    print("🚀 Starting MOCK Fourier GR2 Skill Server on port 8000...")
    mcp.run(transport="streamable-http")
