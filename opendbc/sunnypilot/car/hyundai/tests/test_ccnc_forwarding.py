"""
Copyright (c) 2021-, Haibin Wen, sunnypilot, and a number of other contributors.

This file is part of sunnypilot and is licensed under the MIT License.
See the LICENSE.md file in the root directory for more details.
"""
import unittest

from opendbc.can import CANPacker, CANParser
from opendbc.car import Bus, structs
from opendbc.car.can_definitions import CanData
from opendbc.car.car_helpers import interfaces
from opendbc.car.hyundai import hyundaicanfd
from opendbc.car.hyundai.hyundaicanfd import CanBus
from opendbc.car.hyundai.values import CAR, DBC, HyundaiFlags

MDPS_ADDR = 0xEA
CCNC_161_ADDR = 0x161
CCNC_162_ADDR = 0x162
DT_NANOS = 10_000_000  # one control frame


class TestCcncForwarding(unittest.TestCase):
  """CCNC (non-HDA2) cars: the MDPS copy sent to the camera and the camera's cluster messages copied to E-CAN
  must keep continuous counters regardless of how the received frames are batched per control frame."""

  def setUp(self):
    car_name = CAR.HYUNDAI_TUCSON_2025
    CarInterface = interfaces[car_name]
    CP = CarInterface.get_non_essential_params(car_name)
    CP_SP = CarInterface.get_non_essential_params_sp(CP, car_name)
    self.assertTrue(CP.flags & HyundaiFlags.CCNC and not CP.flags & HyundaiFlags.CANFD_LKA_STEER_MSG)
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
    # nothing is forwarded until the first EPS frame arrived
    self.assertEqual(self._sent(self._step([]), MDPS_ADDR, self.CAN.CAM), [])

    copies = []
    batches = [[self._mdps(MDPS_StrTqSnsrVal=1), self._mdps(MDPS_StrTqSnsrVal=2)], [], [self._mdps(MDPS_StrTqSnsrVal=3)], [], []]
    for frames in batches:
      can_sends = self._step(frames)
      sent = self._sent(can_sends, MDPS_ADDR, self.CAN.CAM)
      self.assertEqual(len(sent), 1)  # one copy per control frame, whatever the batch size
      self.assertIn(MDPS_ADDR, self._decode(self.cam_rx, sent))
      copies.append(dict(self.cam_rx.vl["MDPS"]))

    counters = [c["COUNTER"] for c in copies]
    # the first copy continues the EPS sequence the camera was receiving, then the stream never skips or repeats
    self.assertEqual(counters[0], (batches[0][-1].dat[2] + 1) % 256)
    self.assertEqual(counters, [(counters[0] + i) % 256 for i in range(len(counters))])
    # the last received frame's content is forwarded, with the LKA torque interface reported inactive/unavailable
    self.assertEqual([c["MDPS_StrTqSnsrVal"] for c in copies], [2, 2, 3, 3, 3])
    self.assertTrue(all(c["MDPS_LkaToiActvSta"] == 0 and c["MDPS_LkaToiUnblSta"] == 1 for c in copies))

  def test_ccnc_cluster_messages_forwarded_once_per_frame(self):
    for frames, expected in [([self._ccnc("CCNC_0x161")], 1),
                             ([], 0),
                             ([self._ccnc("CCNC_0x161"), self._ccnc("CCNC_0x161")], 2),
                             ([], 0)]:
      can_sends = self._step(frames)
      sent = self._sent(can_sends, CCNC_161_ADDR, self.CAN.ECAN)
      self.assertEqual(len(sent), expected)
      self.assertEqual(self._sent(can_sends, CCNC_162_ADDR, self.CAN.ECAN), [])
      if expected:
        self.assertIn(CCNC_161_ADDR, self._decode(self.ecan_rx, sent))
        # the camera's own counters are forwarded, in order (COUNTER is byte 2 of the frame)
        self.assertEqual(self.ecan_rx.vl_all["CCNC_0x161"]["COUNTER"], [frame.dat[2] for frame in frames])

    can_sends = self._step([self._ccnc("CCNC_0x162")])
    sent = self._sent(can_sends, CCNC_162_ADDR, self.CAN.ECAN)
    self.assertEqual(len(sent), 1)
    self.assertIn(CCNC_162_ADDR, self._decode(self.ecan_rx, sent))
    self.assertEqual(self.ecan_rx.vl["CCNC_0x162"]["COUNTER"], self.ccnc_counter)

  def test_nothing_sent_without_camera_frames(self):
    for _ in range(50):
      can_sends = self._step([self._mdps()])
      self.assertEqual(self._sent(can_sends, CCNC_161_ADDR, self.CAN.ECAN), [])
      self.assertEqual(self._sent(can_sends, CCNC_162_ADDR, self.CAN.ECAN), [])

  def _hud(self, lat_active, mads_enabled):
    CC = structs.CarControl()
    CC.latActive = lat_active
    CC_SP = structs.CarControlSP()
    CC_SP.mads.available = True
    CC_SP.mads.enabled = mads_enabled
    sent = self._sent(self._step([self._ccnc("CCNC_0x161")], CC, CC_SP), CCNC_161_ADDR, self.CAN.ECAN)
    self.assertEqual(len(sent), 1)
    self.assertIn(CCNC_161_ADDR, self._decode(self.ecan_rx, sent))
    return dict(self.ecan_rx.vl["CCNC_0x161"])

  def test_hud_icons_follow_mads_state(self):
    # steering: green icon, centerline and lane lines (hidden here, the test hud reports no visible lanes)
    hud = self._hud(lat_active=True, mads_enabled=True)
    self.assertEqual((hud["LFA_ICON"], hud["CENTERLINE"], hud["LANELINE_LEFT"], hud["LANELINE_RIGHT"]), (2, 1, 1, 1))
    self.assertEqual((hud["LCA_LEFT_ICON"], hud["LCA_RIGHT_ICON"]), (0, 0))

    # lateral drops out: white blink for a second, then gray while MADS stays enabled, without lane graphics
    hud = self._hud(lat_active=False, mads_enabled=True)
    self.assertEqual((hud["LFA_ICON"], hud["CENTERLINE"], hud["LANELINE_LEFT"], hud["LANELINE_RIGHT"]), (3, 0, 0, 0))
    for _ in range(105):
      CC_SP = structs.CarControlSP()
      CC_SP.mads.available = True
      CC_SP.mads.enabled = True
      self._step([], structs.CarControl(), CC_SP)
    hud = self._hud(lat_active=False, mads_enabled=True)
    self.assertEqual((hud["LFA_ICON"], hud["CENTERLINE"], hud["LANELINE_LEFT"], hud["LANELINE_RIGHT"]), (1, 0, 0, 0))

    # MADS off: icon hidden
    hud = self._hud(lat_active=False, mads_enabled=False)
    self.assertEqual((hud["LFA_ICON"], hud["CENTERLINE"]), (0, 0))

  def test_lane_change_assist_icons_need_blindspot_monitor(self):
    cam_161 = CANParser(DBC[CAR.HYUNDAI_TUCSON_2025][Bus.pt], [("CCNC_0x161", 20)], self.CAN.CAM)
    cam_161.update([(self.now, [self._ccnc("CCNC_0x161")])])
    hud = structs.CarControl.HUDControl()
    out = structs.CarState()
    out.vEgo = 15.0
    for has_bsm, expected in ((False, 0), (True, 4)):
      msg = hyundaicanfd.create_ccnc_161(self.packer, self.CAN, False, False, hud, False, False, dict(cam_161.vl["CCNC_0x161"]), True, out,
                                        False, 2, has_bsm)
      self.assertIn(CCNC_161_ADDR, self._decode(self.ecan_rx, [msg]))
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
