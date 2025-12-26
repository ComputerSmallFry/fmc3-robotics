#!/usr/bin/env python
import argparse
import asyncio
from typing import Any, Dict

from mcp import ClientSession
from mcp.client.streamable_http import streamablehttp_client


async def call_tool(session: ClientSession, name: str, args: Dict[str, Any]) -> Any:
    result = await session.call_tool(name, args)
    return result


async def main() -> None:
    parser = argparse.ArgumentParser(description="SO101 MCP skill smoke test")
    parser.add_argument("--url", default="http://127.0.0.1:8000", help="Skill server base URL")
    parser.add_argument(
        "--robot-config",
        default="configs/lerobot_robot.yaml",
        help="Path to LeRobot robot config YAML",
    )
    parser.add_argument("--camera-name", default="handeye")
    parser.add_argument("--enable-camera", action="store_true", default=True)
    parser.add_argument("--no-camera", dest="enable_camera", action="store_false")
    parser.add_argument("--gripper", action="store_true", help="Open/close gripper during test")
    args = parser.parse_args()

    async with streamablehttp_client(f"{args.url}/mcp") as (read_stream, write_stream, _):
        async with ClientSession(read_stream, write_stream) as session:
            await session.initialize()

            await call_tool(
                session,
                "connect",
                {
                    "robot_config_path": args.robot_config,
                    "camera_name": args.camera_name,
                    "enable_camera": args.enable_camera,
                },
            )

            await call_tool(
                session,
                "get_observation",
                {
                    "image_format": "jpeg_base64",
                    "max_width": 320,
                    "max_height": 240,
                    "include": ["state", "image"] if args.enable_camera else ["state"],
                },
            )

            if args.gripper:
                await call_tool(session, "gripper", {"command": "open", "timeout_s": 1.0})
                await call_tool(session, "gripper", {"command": "close", "timeout_s": 1.0})

            await call_tool(session, "disconnect", {})


if __name__ == "__main__":
    asyncio.run(main())
