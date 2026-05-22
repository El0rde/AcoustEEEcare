#!/usr/bin/env python3
"""
BLE Scanner — lists all nearby BLE devices.
Run this if receiver_new.py can't find AcoustEEEcare.
Compatible with all bleak versions.
"""
import asyncio
from bleak import BleakScanner

async def scan():
    print("Scanning for 10 s — all nearby BLE devices:\n")
    results = await BleakScanner.discover(timeout=10.0, return_adv=True)
    if not results:
        print("  No devices found. Check Bluetooth is on.")
        return
    # results is a dict: {address: (BLEDevice, AdvertisementData)}
    items = sorted(results.values(), key=lambda x: x[1].rssi or -999, reverse=True)
    for device, adv in items:
        print(f"  RSSI={adv.rssi:4d}  {device.address}  name={device.name!r}")
    print(f"\n{len(results)} device(s) found.")

asyncio.run(scan())