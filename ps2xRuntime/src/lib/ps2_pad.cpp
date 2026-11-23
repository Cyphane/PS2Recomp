#include <cstdio>
#include <cstdlib>
#include "runtime/ps2_pad.h"
#include "ps2_host_backend.h"
#include <cstring>
#include <string>
#include <utility>
#include <vector>

namespace
{
    constexpr uint8_t kPadAnalogMarker = 0x73;
    constexpr uint8_t kPadStickCenter = 0x80;

    constexpr uint16_t PAD_LEFT = 0x0080u;
    constexpr uint16_t PAD_DOWN = 0x0040u;
    constexpr uint16_t PAD_RIGHT = 0x0020u;
    constexpr uint16_t PAD_UP = 0x0010u;
    constexpr uint16_t PAD_START = 0x0008u;
    constexpr uint16_t PAD_R3 = 0x0004u;
    constexpr uint16_t PAD_L3 = 0x0002u;
    constexpr uint16_t PAD_SELECT = 0x0001u;
    constexpr uint16_t PAD_SQUARE = 0x8000u;
    constexpr uint16_t PAD_CROSS = 0x4000u;
    constexpr uint16_t PAD_CIRCLE = 0x2000u;
    constexpr uint16_t PAD_TRIANGLE = 0x1000u;
    constexpr uint16_t PAD_R1 = 0x0800u;
    constexpr uint16_t PAD_L1 = 0x0400u;
    constexpr uint16_t PAD_R2 = 0x0200u;
    constexpr uint16_t PAD_L2 = 0x0100u;
}

bool PSPadBackend::readState(int /*port*/, int /*slot*/, uint8_t *data, size_t size)
{
    if (!data || size < 32)
        return false;

    std::memset(data, 0, 32);
    data[0] = 0x01;
    data[1] = kPadAnalogMarker;
    data[2] = 0xFF;
    data[3] = 0xFF;
    data[4] = data[5] = data[6] = data[7] = kPadStickCenter;

    uint16_t btns = 0xFFFFu;
    // First connected gamepad; the keyboard works alongside it.
    int kGamepad = -1;
    for (int i = 0; i < 4 && kGamepad < 0; ++i)
        if (IsGamepadAvailable(i))
            kGamepad = i;
    const bool useGamepad = kGamepad >= 0;
    auto clearBit = [&btns](uint16_t mask)
    { btns &= ~mask; };

    if (useGamepad)
    {
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_UP))
            clearBit(PAD_UP);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_DOWN))
            clearBit(PAD_DOWN);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_LEFT))
            clearBit(PAD_LEFT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_FACE_RIGHT))
            clearBit(PAD_RIGHT);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_DOWN))
            clearBit(PAD_CROSS);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_RIGHT))
            clearBit(PAD_CIRCLE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_LEFT))
            clearBit(PAD_SQUARE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_FACE_UP))
            clearBit(PAD_TRIANGLE);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_1))
            clearBit(PAD_L1);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_1))
            clearBit(PAD_R1);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_TRIGGER_2))
            clearBit(PAD_L2);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_TRIGGER_2))
            clearBit(PAD_R2);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_RIGHT))
            clearBit(PAD_START);
        if (IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_MIDDLE_LEFT))
            clearBit(PAD_SELECT);
        // s3recomp: both sticks clicked together toggle fast-forward and are not passed to the game.
        const bool l3 = IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_LEFT_THUMB);
        const bool r3 = IsGamepadButtonDown(kGamepad, GAMEPAD_BUTTON_RIGHT_THUMB);
        if (l3 && !r3)
            clearBit(PAD_L3);
        if (r3 && !l3)
            clearBit(PAD_R3);

        float lx = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_X);
        float ly = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_LEFT_Y);
        float rx = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_X);
        float ry = GetGamepadAxisMovement(kGamepad, GAMEPAD_AXIS_RIGHT_Y);
        data[6] = static_cast<uint8_t>(128 + lx * 127);
        data[7] = static_cast<uint8_t>(128 + ly * 127);
        data[4] = static_cast<uint8_t>(128 + rx * 127);
        data[5] = static_cast<uint8_t>(128 + ry * 127);
    }
    {
        if (IsKeyDown(KEY_UP) || IsKeyDown(KEY_W))
            clearBit(PAD_UP);
        if (IsKeyDown(KEY_DOWN) || IsKeyDown(KEY_S))
            clearBit(PAD_DOWN);
        if (IsKeyDown(KEY_LEFT) || IsKeyDown(KEY_A))
            clearBit(PAD_LEFT);
        if (IsKeyDown(KEY_RIGHT) || IsKeyDown(KEY_D))
            clearBit(PAD_RIGHT);
        if (IsKeyDown(KEY_X) || IsKeyDown(KEY_SPACE))
            clearBit(PAD_CROSS);
        if (IsKeyDown(KEY_C) || IsKeyDown(KEY_ESCAPE))
            clearBit(PAD_CIRCLE);
        if (IsKeyDown(KEY_Z) || IsKeyDown(KEY_KP_0))
            clearBit(PAD_SQUARE);
        if (IsKeyDown(KEY_V) || IsKeyDown(KEY_KP_1))
            clearBit(PAD_TRIANGLE);
        if (IsKeyDown(KEY_Q))
            clearBit(PAD_L1);
        if (IsKeyDown(KEY_E))
            clearBit(PAD_R1);
        if (IsKeyDown(KEY_LEFT_SHIFT))
            clearBit(PAD_L2);
        if (IsKeyDown(KEY_RIGHT_SHIFT))
            clearBit(PAD_R2);
        if (IsKeyDown(KEY_ENTER))
            clearBit(PAD_START);
        if (IsKeyDown(KEY_TAB))
            clearBit(PAD_SELECT);
    }

    // s3recomp test aid: PS2X_PULSE_START=1 taps Start for a moment every few seconds.
    {
        static const bool pulse = [] { const char *v = std::getenv("PS2X_PULSE_START"); return v && *v == '1'; }();
        if (pulse)
        {
            static uint64_t calls = 0;
            if ((++calls % 400u) < 20u)
                clearBit(PAD_START);
        }
        // PS2X_PULSE_SEQ=START,UP,CROSS,... taps the listed buttons in turn, one every 240 reads.
        static const std::vector<uint16_t> seq = [] {
            std::vector<uint16_t> out;
            const char *v = std::getenv("PS2X_PULSE_SEQ");
            if (!v)
                return out;
            const std::pair<const char *, uint16_t> names[] = {
                {"UP", PAD_UP}, {"DOWN", PAD_DOWN}, {"LEFT", PAD_LEFT}, {"RIGHT", PAD_RIGHT}, {"START", PAD_START},
                {"SELECT", PAD_SELECT}, {"CROSS", PAD_CROSS}, {"CIRCLE", PAD_CIRCLE}, {"SQUARE", PAD_SQUARE},
                {"TRIANGLE", PAD_TRIANGLE}, {"L1", PAD_L1}, {"R1", PAD_R1}, {"L2", PAD_L2}, {"R2", PAD_R2}, {"WAIT", 0}};
            std::string list(v);
            size_t pos = 0;
            while (pos <= list.size())
            {
                size_t end = list.find(',', pos);
                if (end == std::string::npos)
                    end = list.size();
                const std::string name = list.substr(pos, end - pos);
                for (const auto &n : names)
                    if (name == n.first)
                        out.push_back(n.second);
                pos = end + 1;
            }
            return out;
        }();
        // PS2X_PAD_FILE=/path: test aid. Each line written to the file ("CROSS", "UP", ...) is
        // pressed for ~12 reads, in order; the file is truncated once its lines are consumed.
        {
            static const char *padFile = std::getenv("PS2X_PAD_FILE");
            static std::vector<uint16_t> pending;
            static uint64_t fileCalls = 0;
            static uint32_t holdLeft = 0, gapLeft = 0;
            static uint16_t holdMask = 0;
            if (padFile)
            {
                if ((++fileCalls % 10u) == 0u && pending.empty() && holdLeft == 0u)
                {
                    if (FILE *f = std::fopen(padFile, "rb"))
                    {
                        char buf[512] = {};
                        const size_t n = std::fread(buf, 1, sizeof(buf) - 1, f);
                        std::fclose(f);
                        if (n > 0)
                        {
                            const std::pair<const char *, uint16_t> names[] = {
                                {"UP", PAD_UP}, {"DOWN", PAD_DOWN}, {"LEFT", PAD_LEFT}, {"RIGHT", PAD_RIGHT}, {"START", PAD_START},
                                {"SELECT", PAD_SELECT}, {"CROSS", PAD_CROSS}, {"CIRCLE", PAD_CIRCLE}, {"SQUARE", PAD_SQUARE},
                                {"TRIANGLE", PAD_TRIANGLE}, {"L1", PAD_L1}, {"R1", PAD_R1}, {"L2", PAD_L2}, {"R2", PAD_R2}, {"WAIT", 0}};
                            std::string all(buf, n), tok;
                            for (char ch : all + "\n")
                            {
                                if (ch == '\n' || ch == '\r' || ch == ',' || ch == ' ')
                                {
                                    for (const auto &nm : names)
                                        if (tok == nm.first)
                                            pending.push_back(nm.second);
                                    tok.clear();
                                }
                                else
                                    tok += ch;
                            }
                            if (FILE *t = std::fopen(padFile, "wb"))
                                std::fclose(t);
                        }
                    }
                }
                if (holdLeft > 0u)
                {
                    --holdLeft;
                    if (holdMask)
                        clearBit(holdMask);
                    if (holdLeft == 0u)
                        gapLeft = 12u;
                }
                else if (gapLeft > 0u)
                    --gapLeft;
                else if (!pending.empty())
                {
                    holdMask = pending.front();
                    pending.erase(pending.begin());
                    holdLeft = 12u;
                    if (holdMask)
                        clearBit(holdMask);
                }
            }
        }
        if (!seq.empty())
        {
            static uint64_t seqCalls = 0;
            const uint64_t c = seqCalls++;
            if ((c % 240u) < 15u)
                clearBit(seq[(c / 240u) % seq.size()]);
        }
    }

    data[2] = static_cast<uint8_t>(btns & 0xFF);
    data[3] = static_cast<uint8_t>(btns >> 8);
    return true;
}
