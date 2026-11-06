-- AWR2243 + DCA1000 bring-up and waveform helpers for radar_doa_gui.
-- Copied from awr2243-autobahn-labs/configs/awr2243_common.lua (read-only reference).
-- Changes: TX backoff argument in C.profile, configurable band edge (C.MAX_FREQ_GHZ),
-- frame_count 0 allowed in C.advanced_frame (continuous), no PYTHON_* export.
-- Bring-up (connection, firmware, DCA1000, ADC, LVDS) is unchanged.

local C = {}

C.START_FREQ_GHZ = 76.1   -- every ramp ends below C.MAX_FREQ_GHZ
C.MAX_FREQ_GHZ = 77.0     -- 76-77 GHz default; the GUI sets 81.0 for the extended band
C.RX_GAIN = 94            -- encoded RX gain, as in the working TDM/DDM setup
C.NUM_RX = 4
C.TICKS_PER_MS = 200000   -- advanced-frame periods are programmed in 5 ns units
C.MIN_CHIRP_US = 13       -- legacy chirp API minimum cycle (DFP user guide 11.2)
C.SUBFRAME_BLANK_MS = 0.1 -- inter-subframe blank time (DFP user guide 11.1)
C.FRAME_BLANK_MS = 0.2    -- inter-frame blank time (DFP user guide 11.1)
C.MAX_LOOPS = 255
C.MAX_CHIRP_RAM = 512

C.profiles = {}

local unpack = table.unpack or unpack

function C.check(status, name)
    assert(status == 0, name .. " failed: " .. tostring(status))
end
local check = C.check

-- Power up, load firmware, configure DCA1000/LVDS and run RF init (unchanged from the repo).
function C.bringup(per_chirp_phase)
    -- check(ar1.FullReset(), "FullReset")
    check(ar1.SOPControl(2), "SOPControl")
    check(ar1.Connect(3, 115200, 1000), "Connect")

    ar1.SelectChipVersion("XWR2243")
    ar1.deviceVariantSelection("XWR2243")
    ar1.frequencyBandSelection("77G")

    check(ar1.DownloadBSSFw("C:\\ti\\mmwave_dfp_02_02_05_00\\rf_eval\\rf_eval_firmware\\radarss\\xwr2xxx_radarss_rprc.bin"), "DownloadBSSFw")
    check(ar1.DownloadMSSFw("C:\\ti\\mmwave_dfp_02_02_05_00\\rf_eval\\rf_eval_firmware\\masterss\\xwr22xx_masterss_rprc.bin"), "DownloadMSSFw")
    check(ar1.PowerOn(0, 1000, 0, 0), "PowerOn")
    check(ar1.RfEnable(), "RfEnable")

    ar1.SelectCaptureDevice("DCA1000")
    check(ar1.CaptureCardConfig_EthInit("192.168.33.30", "192.168.33.180", "12:34:56:78:90:12", 4096, 4098), "CaptureCardConfig_EthInit")
    check(ar1.CaptureCardConfig_Mode(1, 1, 1, 2, 3, 30), "CaptureCardConfig_Mode")
    check(ar1.CaptureCardConfig_PacketDelay(25), "CaptureCardConfig_PacketDelay")

    -- TX0 + TX2, RX0..3, 16 bit, complex 1x.
    check(ar1.ChanNAdcConfig(1, 0, 1, 1, 1, 1, 1, 2, 1, 0), "ChanNAdcConfig")
    check(ar1.LPModConfig(0, 0), "LPModConfig")
    if per_chirp_phase then
        check(ar1.SetMiscConfig(1, 0, 0, 0), "SetMiscConfig")
    end
    check(ar1.RfInit(), "RfInit")

    check(ar1.DataPathConfig(513, 1216644097, 0), "DataPathConfig")
    check(ar1.LvdsClkConfig(1, 1), "LvdsClkConfig")
    check(ar1.LVDSLaneConfig(0, 1, 1, 1, 1, 1, 0, 0), "LVDSLaneConfig")
end

-- p = {id, idle_us, adc_start_us, ramp_end_us, slope_mhz_us, samples, fs_ksps, backoff_db}
function C.profile(p, max_freq_var_mhz)
    local sampling_us = p.samples / (p.fs_ksps / 1000)
    local top_ghz = C.START_FREQ_GHZ + ((max_freq_var_mhz or 0)
        + p.slope_mhz_us * p.ramp_end_us) / 1000
    local backoff = p.backoff_db or 0
    assert(p.adc_start_us + sampling_us <= p.ramp_end_us,
        "Profile " .. p.id .. ": ADC window exceeds ramp end")
    assert(p.idle_us + p.ramp_end_us >= C.MIN_CHIRP_US,
        "Profile " .. p.id .. ": chirp cycle below 13 us")
    assert(top_ghz <= C.MAX_FREQ_GHZ + 1e-6,
        "Profile " .. p.id .. ": ramp ends above " .. C.MAX_FREQ_GHZ .. " GHz")
    assert(backoff >= 0 and backoff <= 26, "TX backoff must be 0..26 dB")
    -- TX0 and TX2 backoff; TX1 is disabled
    check(ar1.ProfileConfig(p.id, C.START_FREQ_GHZ, p.idle_us, p.adc_start_us,
        p.ramp_end_us, backoff, 0, backoff, 0, 0, 0, p.slope_mhz_us, 0, p.samples,
        p.fs_ksps, 0, 0, C.RX_GAIN), "ProfileConfig " .. p.id)
    C.profiles[p.id] = p
end

-- One chirp-RAM entry. tx0/tx2 are 0/1 enables; phase codes in 5.625 deg units.
function C.chirp(index, profile_id, tx0, tx2, freq_var_mhz, phase0, phase2)
    assert(index < C.MAX_CHIRP_RAM, "Chirp RAM holds 512 entries")
    check(ar1.ChirpConfig(index, index, profile_id, freq_var_mhz or 0, 0, 0, 0,
        tx0, 0, tx2), "ChirpConfig " .. index)
    if phase0 then
        check(ar1.SetPerChirpPhaseShifterConfig(index, index, phase0, 0, phase2),
            "Phase chirp " .. index)
    end
end

local function active_ms(sf)
    local p = C.profiles[sf.profile]
    assert(p, sf.id .. ": profile " .. tostring(sf.profile) .. " not configured")
    return sf.num_chirps * sf.loops * (p.idle_us + p.ramp_end_us) / 1000
end

-- sf = {id, profile, chirp_start, num_chirps, loops}; frame_count 0 = continuous.
function C.advanced_frame(subframes, frame_count, frame_period_ms)
    local n = #subframes
    assert(n >= 1 and n <= 4, "Advanced frame supports 1..4 subframes")
    assert(frame_count >= 0 and frame_count <= 65535 and frame_count % 1 == 0,
        "frame_count must be 0..65535")
    local sf_ms = frame_period_ms / n
    local ticks = math.floor(sf_ms * C.TICKS_PER_MS + 0.5)
    local seq, data = {n, 1536}, {}  -- 1536 as in TI's Studio examples
    for i = 1, n do
        local sf = subframes[i]
        local blank = (i == n) and C.FRAME_BLANK_MS or C.SUBFRAME_BLANK_MS
        assert(sf.loops >= 1 and sf.loops <= C.MAX_LOOPS, sf.id .. ": loops must be 1..255")
        assert(active_ms(sf) + blank <= sf_ms,
            sf.id .. ": chirps do not fit the subframe period")
    end
    for i = 1, 4 do
        local sf = subframes[math.min(i, n)]
        local samples = C.profiles[sf.profile].samples
        for _, v in ipairs({sf.profile, sf.chirp_start, sf.num_chirps, sf.loops,
                            ticks, 0, 1, 1, ticks}) do
            seq[#seq + 1] = v
        end
        for _, v in ipairs({sf.num_chirps * sf.loops, 2 * samples, 1}) do
            data[#data + 1] = v
        end
    end
    for _, v in ipairs({frame_count, 1, 0, n}) do
        seq[#seq + 1] = v
    end
    for _, v in ipairs(data) do
        seq[#seq + 1] = v
    end
    check(ar1.AdvanceFrameConfig(unpack(seq)), "AdvanceFrameConfig")
end

return C
