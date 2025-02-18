"""Network interface to the MABX using Julian Bedei's protocol.

File:   mabx_network_interface.py
Author: Kevin Badalian (badalian_k@mmp.rwth-aachen.de)
        Vasu Sharma(vasu.sharma@rwth-aachen.de)
        Teaching and Research Area Mechatronics in Mobile Propulsion (MMP)
        RWTH Aachen University
Date:   2022-11-08
"""

import sys
import socket
import struct

import logging
from typing import Any


logger = logging.getLogger(__name__)


class MabxNetworkInterface:
    """Network interface that communicates with an MABX using Julian Bedei's
    protocol.
    """

    # Constants
    MSG_SIZE_FROM_MABX = 80
    MSG_SIZE_TO_MABX = 28

    def __init__(
        self, pi_addr: str, pi_port: int, mabx_addr: str, mabx_port: int
    ) -> None:
        """Initialize the interface.

        Arguments:
          - pi_addr: str
              IP address of the Raspberry Pi.
          - pi_port: int
              Port for listening.
          - mabx_addr: str
              IP address of the MABX.
          - mabx_port: int
              Listening port of the MABX.
        """

        self._sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self._sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self._sock.bind((pi_addr, pi_port))
        self._mabx_addr = mabx_addr
        self._mabx_port = mabx_port

    def terminate(self) -> None:
        """Shut the interface down."""

        if self._sock is not None:
            self._sock.close()
            self._sock = None

    def send(self, msg: dict[str, Any]) -> None:
        """Send a message.

        Arguments:
          - msg: dict[str, float]
              Dictionary containing the message to send.

        Raises:
          - A `ValueError` is raised if `msg` doesn't contain all required
            fields.
        """

        if "DOIMain" not in msg or "bStrtReq" not in msg:
            raise ValueError(
                "The message dictionary doesn't contain all required" " fields."
            )

        data = b""
        # data += socket.htonl(int.from_bytes(struct.pack("f", msg["NVO"]),
        #    byteorder=sys.byteorder)).to_bytes(4, byteorder=sys.byteorder)
        data += int.from_bytes(
            struct.pack("f", msg["DOIMain"]), byteorder=sys.byteorder
        ).to_bytes(4, byteorder=sys.byteorder)
        data += int.from_bytes(
            struct.pack("f", msg["P2M"]), byteorder=sys.byteorder
        ).to_bytes(4, byteorder=sys.byteorder)
        data += int.from_bytes(
            struct.pack("f", msg["SOIMain"]), byteorder=sys.byteorder
        ).to_bytes(4, byteorder=sys.byteorder)
        data += int.from_bytes(
            struct.pack("f", msg["DOIH2"]), byteorder=sys.byteorder
        ).to_bytes(4, byteorder=sys.byteorder)
        data += int.from_bytes(
            struct.pack("f", msg["CycleCounter"]), byteorder=sys.byteorder
        ).to_bytes(4, byteorder=sys.byteorder)
        data += int.from_bytes(
            struct.pack("f", msg["bStrtReq"]), byteorder=sys.byteorder
        ).to_bytes(4, byteorder=sys.byteorder)
        data += int.from_bytes(
            struct.pack("f", msg["bValidationCycle"]), byteorder=sys.byteorder
        ).to_bytes(4, byteorder=sys.byteorder)
        # flags = 0
        # if msg["bStrtReq"]:
        #     flags |= 1 << 0
        # if msg["bValidationCycle"]:
        #     flags |= 1 << 1
        # # data += socket.htonl(flags).to_bytes(4, byteorder=sys.byteorder)
        # data += flags.to_bytes(1, byteorder=sys.byteorder)

        self._sock.sendto(data, (self._mabx_addr, self._mabx_port))

    def recv(self) -> dict[str, Any]:
        """Receive a message.

        Returns:
          - _: dict[str, float]
              Dictionary containing the converted values of the received
              message.

        Raises:
          - RuntimeError
              A `RuntimeError` is raised if the sender is unknown or if the UDP
              packet doesn't have the correct size.
        """

        # Receive data and perform checks
        data, addr = self._sock.recvfrom(
            MabxNetworkInterface.MSG_SIZE_FROM_MABX
        )


        if addr[0] != self._mabx_addr or addr[1] != self._mabx_port:
            logger.info("UDP packet is from an unknown sender.")
            raise RuntimeError("Received a UDP packet from an unknown sender.")
        if len(data) != MabxNetworkInterface.MSG_SIZE_FROM_MABX:
            logger.info("Received UDP packet doesn't have the right size.")
            raise RuntimeError(
                "Receuved UDP packet doesn't have the right size."
            )



        # Extract the received quantities
        d = {}
        # Extract numeric values
        # print(data)
        d["IMEPLast"] = struct.unpack("<f", data[0:4])
        d["NOXLast"] = struct.unpack("<f", data[4:8])
        d["DeltaNOx"] = struct.unpack("<f", data[8:12])
        d["MPRR"] = struct.unpack("<f", data[12:16])
        d["IMEPRefLast"] = struct.unpack("<f", data[16:20])
        d["IMEPRef"] = struct.unpack("<f", data[20:24])
        d["DeltaIMEP"] = struct.unpack("<f", data[24:28])
        d["ErrorIMEP"] = struct.unpack("<f", data[28:32])
        d["Reward"] = struct.unpack("<f", data[32:36])
        d["MABXCycleCounter"] = struct.unpack("<f", data[36:40])
        # Extract flags
        flags = int.from_bytes(data[40:44], byteorder="little")
        d["bMABX Ready"] = bool(flags & (1 << 0))
        d["bTerminateEpisode"] = bool(flags & (1 << 1))
        #print(f'Terminate Episode: {d["bTerminateEpisode"]}\nMABX Ready: {d["bMABX Ready"]}')
        d["OpState"] = struct.unpack("<f", data[44:48])
        d["HiddenState1"] = struct.unpack("<f", data[48:52])
        d["HiddenState2"] = struct.unpack("<f", data[52:56])
        d["HiddenState3"] = struct.unpack("<f", data[56:60])
        d["HiddenState4"] = struct.unpack("<f", data[60:64])
        d["HiddenState5"] = struct.unpack("<f", data[64:68])
        d["HiddenState6"] = struct.unpack("<f", data[68:72])
        d["HiddenState7"] = struct.unpack("<f", data[72:76])
        d["HiddenState8"] = struct.unpack("<f", data[76:80])
        return d
