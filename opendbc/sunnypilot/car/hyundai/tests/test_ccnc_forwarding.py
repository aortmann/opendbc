"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, gen_empty_fingerprint, structs
from opendbc.car.can_definitions import CanData
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai.hyundaicanfd import CanBus
from opendbc.car.hyundai.values import CAR, DBC, HyundaiFlags
from opendbc.sunnypilot.car.hyundai import ccnc
from opendbc.sunnypilot.car.hyundai.values import HyundaiFlagsSP

MDPS_ADDR = 0xEA
CCNC_161_ADDR = 0x161
CCNC_162_ADDR = 0x162
DT_NANOS = 10_000_000  # one control frame
CLUSTER_STEP = 5  # the cluster messages are sent every 5 control frames


class TestCcncForwarding(unittest.TestCase):
  """CCNC (non-HDA2) cars: the MDPS copy sent to the camera and the cluster messages sent to E-CAN
  must keep continuous counters regardless of how the received frames are batched per control frame."""

  def setUp(self):
    car_name = CAR.HYUNDAI_TUCSON_2025
    CarInterface = interfaces[car_name]
    # ccNC is detected from the camera's cluster messages
    fingerprint = gen_empty_fingerprint()
    cam_bus = CanBus(None, fingerprint).CAM
    fingerprint[cam_bus].update({CCNC_161_ADDR: 32, CCNC_162_ADDR: 32})
    CP = CarInterface.get_params(car_name, fingerprint, [], False, False, False)
    CP_SP = CarInterface.get_params_sp(CP, car_name, fingerprint, [], False, False, False)
    self.assertTrue(CP_SP.flags & HyundaiFlagsSP.CCNC and not CP.flags & HyundaiFlags.CANFD_LKA_STEER_MSG)
    self.assertFalse(CP.flags & HyundaiFlags.HAS_BSM)

    self.CI = CarInterface(CP, CP_SP)
    self.CAN = CanBus(CP)
    self.packer = CANPacker(DBC[car_name][Bus.pt])
    self.cam_rx = CANParser(DBC[car_name][Bus.pt], [("MDPS", 100)], self.CAN.CAM)  # what the camera sees on its bus
    self.ecan_rx = CANParser(DBC[car_name][Bus.pt], [("CCNC_0x161", 20), ("CCNC_0x162", 20)], self.CAN.ECAN)  # what the cluster sees
    self.now = 0
    self.mdps_counter = 0
    self.ccnc_counter = 0
    self.CS = None
    # the car parsers register messages on first use, so run one empty frame before feeding data
    self._step([])

  def _mdps(self, **values):
    self.mdps_counter = (self.mdps_counter + 1) % 256
    addr, dat, bus = self.packer.make_can_msg("MDPS", self.CAN.ECAN, {"COUNTER": self.mdps_counter, "MDPS_LkaToiActvSta": 1, **values})
    return CanData(addr, dat, bus)

  def _ccnc(self, msg):
    self.ccnc_counter = (self.ccnc_counter + 1) % 256
    addr, dat, bus = self.packer.make_can_msg(msg, self.CAN.CAM, {"COUNTER": self.ccnc_counter})
    return CanData(addr, dat, bus)

  def _step(self, frames, CC=None, CC_SP=None):
    self.now += DT_NANOS
    self.CS, _ = self.CI.update([(self.now, frames)])
    CC = structs.CarControl() if CC is None else CC
    CC_SP = structs.CarControlSP() if CC_SP is None else CC_SP
    _, can_sends = self.CI.apply(CC.as_reader(), CC_SP, self.now)
    return can_sends

  def _sent(self, can_sends, addr, bus):
    return [CanData(*msg) for msg in can_sends if msg[0] == addr and msg[2] == bus]

  def _decode(self, parser, can_sends):
    # feed openpilot's output through a real parser: a frame with a bad checksum is not reported as updated
    return parser.update([(self.now, can_sends)])

  def test_mdps_copy_counter_is_continuous(self):
    copies = []
    batches = [[self._mdps(MDPS_StrTqSnsrVal=1), self._mdps(MDPS_StrTqSnsrVal=2)], [], [self._mdps(MDPS_StrTqSnsrVal=3)], [], []]
    for frames in batches:
      can_sends = self._step(frames)
      sent = self._sent(can_sends, MDPS_ADDR, self.CAN.CAM)
      self.assertEqual(len(sent), 1)  # one copy per control frame, whatever the batch size
      self.assertTrue(MDPS_ADDR in self._decode(self.cam_rx, sent))
      copies.append(dict(self.cam_rx.vl["MDPS"]))

    counters = [c["COUNTER"] for c in copies]
    # the copy carries its own counter, so the stream never skips or repeats
    self.assertEqual(counters, [(counters[0] + i) % 256 for i in range(len(counters))])
    # the last received frame's content is forwarded, with the LKA torque interface reported inactive/unavailable
    self.assertEqual([c["MDPS_StrTqSnsrVal"] for c in copies], [2, 2, 3, 3, 3])
    self.assertTrue(all(c["MDPS_LkaToiActvSta"] == 0 and c["MDPS_LkaToiUnblSta"] == 1 for c in copies))

  def test_ccnc_cluster_messages_keep_their_own_counters(self):
    # the camera frames arrive in uneven batches; the copies go out at a fixed rate with their own counters
    counters: dict[int, list[int]] = {CCNC_161_ADDR: [], CCNC_162_ADDR: []}
    sent_per_frame = []
    for i in range(10 * CLUSTER_STEP):
      frames = [self._ccnc("CCNC_0x161"), self._ccnc("CCNC_0x162")] * (i % 3)
      can_sends = self._step(frames)
      sent_per_frame.append(len(self._sent(can_sends, CCNC_161_ADDR, self.CAN.ECAN)))
      for addr, msg in ((CCNC_161_ADDR, "CCNC_0x161"), (CCNC_162_ADDR, "CCNC_0x162")):
        sent = self._sent(can_sends, addr, self.CAN.ECAN)
        self.assertLessEqual(len(sent), 1)
        if sent:
          self.assertTrue(addr in self._decode(self.ecan_rx, sent))
          counters[addr].append(int(self.ecan_rx.vl[msg]["COUNTER"]))

    self.assertEqual(sum(sent_per_frame), 10)
    for values in counters.values():
      self.assertEqual(len(values), 10)
      self.assertEqual(values, [(values[0] + i) % 256 for i in range(len(values))])

  def _hud(self, lat_active, mads_enabled):
    CC = structs.CarControl()
    CC.latActive = lat_active
    CC_SP = structs.CarControlSP()
    CC_SP.mads.available = True
    CC_SP.mads.enabled = mads_enabled
    sent = []
    for _ in range(CLUSTER_STEP):
      sent += self._sent(self._step([self._ccnc("CCNC_0x161")], CC, CC_SP), CCNC_161_ADDR, self.CAN.ECAN)
    self.assertEqual(len(sent), 1)
    self.assertTrue(CCNC_161_ADDR in self._decode(self.ecan_rx, sent))
    return dict(self.ecan_rx.vl["CCNC_0x161"])

  def test_hud_icons_follow_mads_state(self):
    # steering: green icon, centerline and lane lines; no lane change assist icons without blind spot monitoring
    hud = self._hud(lat_active=True, mads_enabled=True)
    self.assertEqual((hud["LFA_ICON"], hud["CENTERLINE"], hud["LANELINE_LEFT"], hud["LANELINE_RIGHT"]), (2, 1, 6, 6))
    self.assertEqual((hud["LCA_LEFT_ICON"], hud["LCA_RIGHT_ICON"]), (0, 0))

    # lateral paused while MADS stays enabled: gray icon, without lane graphics
    hud = self._hud(lat_active=False, mads_enabled=True)
    self.assertEqual((hud["LFA_ICON"], hud["CENTERLINE"], hud["LANELINE_LEFT"], hud["LANELINE_RIGHT"]), (1, 0, 0, 0))

    # MADS off: icon hidden
    hud = self._hud(lat_active=False, mads_enabled=False)
    self.assertEqual((hud["LFA_ICON"], hud["CENTERLINE"]), (0, 0))

  def test_lane_change_assist_icons_need_blindspot_monitor(self):
    cam_161 = CANParser(DBC[CAR.HYUNDAI_TUCSON_2025][Bus.pt], [("CCNC_0x161", 20)], self.CAN.CAM)
    cam_161.update([(self.now, [self._ccnc("CCNC_0x161")])])
    hud = structs.CarControl.HUDControl()
    for has_bsm, expected in ((False, 0), (True, 2)):
      msg = ccnc.create_ccnc_0x161(self.packer, self.CAN, 0, dict(cam_161.vl["CCNC_0x161"]), False, True, False, False, False, False,
                                   hud, 0, 15.0, False, False, has_bsm)
      self.assertTrue(CCNC_161_ADDR in self._decode(self.ecan_rx, [msg]))
      self.assertEqual((self.ecan_rx.vl["CCNC_0x161"]["LCA_LEFT_ICON"], self.ecan_rx.vl["CCNC_0x161"]["LCA_RIGHT_ICON"]), (expected, expected))

  def test_steering_pressed_debounce(self):
    # CCNC: the driver's hands must be above the threshold for 100 ms before an override is reported
    pressed = []
    for _ in range(12):
      self._step([self._mdps(MDPS_StrTqSnsrVal=300)])
      pressed.append(self.CS.steeringPressed)
    self.assertEqual(pressed[:10], [False] * 10)
    self.assertTrue(pressed[10])

  def test_cruise_speed_unset(self):
    # VSetDis 255 is the camera's "no set speed": report 0 like the other brands do
    addr, dat, bus = self.packer.make_can_msg("SCC_CONTROL", self.CAN.CAM, {"VSetDis": 255})
    self._step([CanData(addr, dat, bus)])
    self.assertEqual(self.CS.cruiseState.speed, 0)
    addr, dat, bus = self.packer.make_can_msg("SCC_CONTROL", self.CAN.CAM, {"VSetDis": 90})
    self._step([CanData(addr, dat, bus)])
    self.assertAlmostEqual(self.CS.cruiseState.speed, 90 / 3.6, places=3)


if __name__ == "__main__":
  unittest.main()
