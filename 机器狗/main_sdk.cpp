#include <fcntl.h>
#include <termios.h>
#include <unistd.h>

#include <bitset>
#include <cstdint>
#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <array>
#include <cstring>
#include <exception>
#include <sstream>
#include <string>
#include <thread>
#include <vector>

#include "/usr/local/include/zsibot/include/zsibot_api.h"

using namespace zsibot;

struct RobotConfig
{
    std::string dog_ip = "192.168.234.1";
    uint32_t send_port = 8081;
    uint32_t recv_port = 8080;
};

static std::string json_escape(const std::string &input)
{
    std::ostringstream oss;
    for (unsigned char ch : input)
    {
        switch (ch)
        {
            case '\"':
                oss << "\\\"";
                break;
            case '\\':
                oss << "\\\\";
                break;
            case '\b':
                oss << "\\b";
                break;
            case '\f':
                oss << "\\f";
                break;
            case '\n':
                oss << "\\n";
                break;
            case '\r':
                oss << "\\r";
                break;
            case '\t':
                oss << "\\t";
                break;
            default:
                if (ch < 0x20)
                {
                    oss << "\\u"
                        << std::uppercase << std::hex << std::setw(4) << std::setfill('0')
                        << static_cast<int>(ch) << std::nouppercase << std::dec;
                }
                else
                {
                    oss << ch;
                }
                break;
        }
    }
    return oss.str();
}

static std::string json_string(const std::string &value)
{
    return "\"" + json_escape(value) + "\"";
}

template <typename T>
static void append_json_field(std::ostringstream &oss, bool &first, const std::string &key, const T &value)
{
    if (!first)
    {
        oss << ",";
    }
    first = false;
    oss << json_string(key) << ":" << value;
}

static void append_json_field(std::ostringstream &oss, bool &first, const std::string &key, const std::string &value)
{
    if (!first)
    {
        oss << ",";
    }
    first = false;
    oss << json_string(key) << ":" << json_string(value);
}

static void append_json_field(std::ostringstream &oss, bool &first, const std::string &key, const char *value)
{
    append_json_field(oss, first, key, std::string(value ? value : ""));
}

static void append_json_field(std::ostringstream &oss, bool &first, const std::string &key, bool value)
{
    if (!first)
    {
        oss << ",";
    }
    first = false;
    oss << json_string(key) << ":" << (value ? "true" : "false");
}

static void append_json_field(std::ostringstream &oss, bool &first, const std::string &key, const std::vector<std::string> &values)
{
    if (!first)
    {
        oss << ",";
    }
    first = false;
    oss << json_string(key) << ":[";
    for (size_t i = 0; i < values.size(); ++i)
    {
        if (i > 0)
        {
            oss << ",";
        }
        oss << json_string(values[i]);
    }
    oss << "]";
}

static std::string make_faults_json(const std::vector<FaultInfo> &faults)
{
    std::ostringstream oss;
    oss << "[";
    for (size_t i = 0; i < faults.size(); ++i)
    {
        if (i > 0)
        {
            oss << ",";
        }
        const auto &fault = faults[i];
        oss << "{";
        bool first = true;
        append_json_field(oss, first, "module", fault.module);
        append_json_field(oss, first, "submodule", fault.submodule);
        append_json_field(oss, first, "error_code", fault.error_code);
        append_json_field(oss, first, "level", fault.level);
        append_json_field(oss, first, "info", fault.info);
        oss << "}";
    }
    oss << "]";
    return oss.str();
}

static std::string make_status_json(const ZsibotExecutor &exec)
{
    std::ostringstream oss;
    oss << "{";
    bool first = true;

    append_json_field(oss, first, "connected", exec.IsConnected());
    append_json_field(oss, first, "sn", exec.GetSn());
    append_json_field(oss, first, "dev_name", exec.GetDevName());
    append_json_field(oss, first, "wifi_ssid", exec.GetWifiSsid());
    append_json_field(oss, first, "power", exec.GetPower());
    append_json_field(oss, first, "temperature", exec.GetTemperature());
    append_json_field(oss, first, "speed_level", static_cast<int>(exec.GetSpeedLevel()));
    append_json_field(oss, first, "function_mode", static_cast<int>(exec.GetFunctionMode()));
    append_json_field(oss, first, "control_mode", static_cast<int>(exec.GetControlMode()));
    append_json_field(oss, first, "motion_mode", static_cast<int>(exec.GetMotionMode()));
    append_json_field(oss, first, "motion_type", static_cast<int>(exec.GetMotionType()));
    append_json_field(oss, first, "model", static_cast<int>(exec.GetModel()));

    SpeedInfo speed = exec.GetSpeed();
    append_json_field(oss, first, "speed", speed.speed);
    append_json_field(oss, first, "angle_speed", speed.angle_speed);
    append_json_field(oss, first, "shift_speed", speed.shift_speed);
    append_json_field(oss, first, "angle", speed.angle);

    const auto motor_temps = exec.GetMotorTemp();
    append_json_field(oss, first, "motor_temp0", motor_temps[0]);
    append_json_field(oss, first, "motor_temp1", motor_temps[1]);
    append_json_field(oss, first, "motor_temp2", motor_temps[2]);
    append_json_field(oss, first, "motor_temp3", motor_temps[3]);

    const auto battery = exec.GetBatteryInfo();
    oss << ",\"battery\":{";
    bool battery_first = true;
    append_json_field(oss, battery_first, "current", battery.current);
    append_json_field(oss, battery_first, "error", battery.error);
    append_json_field(oss, battery_first, "power", battery.power);
    append_json_field(oss, battery_first, "temp", battery.temp);
    append_json_field(oss, battery_first, "volt", battery.volt);
    oss << "}";

    const auto version = exec.GetVersion();
    oss << ",\"version\":{";
    bool version_first = true;
    append_json_field(oss, version_first, "mc_version", version.mc_version);
    append_json_field(oss, version_first, "dog_task_version", version.dog_task_version);
    oss << "}";

    const std::vector<FaultInfo> faults = exec.GetFaultInfo();
    append_json_field(oss, first, "fault_count", static_cast<uint32_t>(faults.size()));
    oss << ",\"faults\":" << make_faults_json(faults);

    oss << "}";
    return oss.str();
}

static void send_json_response(bool ok, const std::string &command, const std::string &message = "",
                               const std::string &data_json = "")
{
    std::ostringstream oss;
    oss << "{\"ok\":" << (ok ? "true" : "false") << ",\"command\":" << json_string(command);
    if (!message.empty())
    {
        oss << ",\"message\":" << json_string(message);
    }
    if (!data_json.empty())
    {
        oss << ",\"data\":" << data_json;
    }
    oss << "}";
    std::cout << oss.str() << std::endl;
}

static bool parse_u32(const std::string &value, uint32_t &out)
{
    try
    {
        size_t idx = 0;
        const unsigned long parsed = std::stoul(value, &idx, 10);
        if (idx != value.size() || parsed > 0xFFFFFFFFul)
        {
            return false;
        }
        out = static_cast<uint32_t>(parsed);
        return true;
    }
    catch (...)
    {
        return false;
    }
}

static void print_usage(const char *prog)
{
    std::cerr << "用法: " << prog
              << " [--daemon] [--dog-ip IP] [--send-port PORT] [--recv-port PORT]\n"
              << "默认: dog-ip=192.168.234.1 send-port=8081 recv-port=8080\n";
}

static bool parse_args(int argc, char *argv[], bool &daemon_mode, RobotConfig &config, bool &help_requested)
{
    daemon_mode = false;
    help_requested = false;
    for (int i = 1; i < argc; ++i)
    {
        std::string arg = argv[i];
        auto take_value = [&](std::string &target) -> bool {
            auto eq = arg.find('=');
            if (eq != std::string::npos)
            {
                target = arg.substr(eq + 1);
                return true;
            }
            if (i + 1 >= argc)
            {
                return false;
            }
            std::string next = argv[i + 1];
            if (!next.empty() && next[0] == '-')
            {
                return false;
            }
            target = argv[++i];
            return true;
        };

        if (arg == "--daemon")
        {
            daemon_mode = true;
        }
        else if (arg == "--help" || arg == "-h")
        {
            print_usage(argv[0]);
            help_requested = true;
            return true;
        }
        else if (arg == "--dog-ip" || arg.rfind("--dog-ip=", 0) == 0)
        {
            if (!take_value(config.dog_ip))
            {
                std::cerr << "缺少 --dog-ip 参数值\n";
                return false;
            }
            if (config.dog_ip.empty())
            {
                std::cerr << "无效的 --dog-ip 参数\n";
                return false;
            }
        }
        else if (arg == "--send-port" || arg.rfind("--send-port=", 0) == 0)
        {
            std::string value;
            if (!take_value(value) || !parse_u32(value, config.send_port))
            {
                std::cerr << "无效的 --send-port 参数\n";
                return false;
            }
            if (config.send_port == 0)
            {
                std::cerr << "无效的 --send-port 参数\n";
                return false;
            }
        }
        else if (arg == "--recv-port" || arg.rfind("--recv-port=", 0) == 0)
        {
            std::string value;
            if (!take_value(value) || !parse_u32(value, config.recv_port))
            {
                std::cerr << "无效的 --recv-port 参数\n";
                return false;
            }
            if (config.recv_port == 0)
            {
                std::cerr << "无效的 --recv-port 参数\n";
                return false;
            }
        }
        else
        {
            std::cerr << "未知参数: " << arg << "\n";
            print_usage(argv[0]);
            return false;
        }
    }
    return true;
}

void set_conio_terminal_mode()
{
    struct termios new_termios;
    tcgetattr(STDIN_FILENO, &new_termios);
    new_termios.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &new_termios);
}

void reset_terminal_mode()
{
    struct termios old_termios;
    tcgetattr(STDIN_FILENO, &old_termios);
    old_termios.c_lflag |= (ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &old_termios);
}

int kbhit()
{
    struct termios oldt, newt;
    int oldf;
    tcgetattr(STDIN_FILENO, &oldt);
    newt = oldt;
    newt.c_lflag &= ~(ICANON | ECHO);
    tcsetattr(STDIN_FILENO, TCSANOW, &newt);
    oldf = fcntl(STDIN_FILENO, F_GETFL, 0);
    fcntl(STDIN_FILENO, F_SETFL, oldf | O_NONBLOCK);

    int ch = getchar();

    tcsetattr(STDIN_FILENO, TCSANOW, &oldt);
    fcntl(STDIN_FILENO, F_SETFL, oldf);

    if (ch != EOF)
    {
        ungetc(ch, stdin);
        return 1;
    }

    return 0;
}

void daemon_mode(const RobotConfig &config)
{
    ZsibotExecutor zsibot_exec(Role::ROLE_SDK, config.dog_ip, config.send_port, config.recv_port);

    std::string line;
    while (std::getline(std::cin, line))
    {
        if (!line.empty() && line.back() == '\r')
        {
            line.pop_back();
        }
        if (line.empty())
        {
            continue;
        }

        try
        {
            if (line == "forward")
            {
                zsibot_exec.SetRemote({0.1f, 0, 0, 0}, std::array<float32_t, 14>{0});
                send_json_response(true, "forward");
            }
            else if (line == "backward")
            {
                zsibot_exec.SetRemote({-0.1f, 0, 0, 0}, std::array<float32_t, 14>{0});
                send_json_response(true, "backward");
            }
            else if (line == "turn_left")
            {
                zsibot_exec.SetRemote({0, 0, 0.04f, 0}, std::array<float32_t, 14>{0});
                send_json_response(true, "turn_left");
            }
            else if (line == "turn_right")
            {
                zsibot_exec.SetRemote({0, 0, -0.05f, 0}, std::array<float32_t, 14>{0});
                send_json_response(true, "turn_right");
            }
            else if (line == "move_left")
            {
                zsibot_exec.SetRemote({0, 0.04f, 0, 0}, std::array<float32_t, 14>{0});
                send_json_response(true, "move_left");
            }
            else if (line == "move_right")
            {
                zsibot_exec.SetRemote({0, -0.02f, 0, 0}, std::array<float32_t, 14>{0});
                send_json_response(true, "move_right");
            }
            else if (line == "stop_move")
            {
                zsibot_exec.SetRemote({0, 0, 0, 0}, std::array<float32_t, 14>{0});
                send_json_response(true, "stop_move");
            }
            else if (line == "emergency_stop")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_EMERGENCY_STOP);
                send_json_response(true, "emergency_stop");
            }
            else if (line == "sit_down")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_SIT_DOWN);
                send_json_response(true, "sit_down");
            }
            else if (line == "stand_up")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_STAND_UP);
                send_json_response(true, "stand_up");
            }
            else if (line == "jump")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_JUMP);
                send_json_response(true, "jump");
            }
            else if (line == "forward_jump")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_FORWARD_JUMP);
                send_json_response(true, "forward_jump");
            }
            else if (line == "greet")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_GREET);
                send_json_response(true, "greet");
            }
            else if (line == "back_flip")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_BACK_FLIP);
                send_json_response(true, "back_flip");
            }
            else if (line == "enter_lab")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_ENTER_LAB_MODE);
                send_json_response(true, "enter_lab");
            }
            else if (line == "exit_lab")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_EXIT_LAB_MODE);
                send_json_response(true, "exit_lab");
            }
            else if (line == "two_leg_stand")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_TWO_LEG_STAND);
                send_json_response(true, "two_leg_stand");
            }
            else if (line == "sdk_mode")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_SDK_CONTROL_RIGHT);
                send_json_response(true, "sdk_mode");
            }
            else if (line == "remote_mode")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_REMOTE_CONTROL_RIGHT);
                send_json_response(true, "remote_mode");
            }
            else if (line == "roamerx_mode")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_ROAMERX_CONTROL_RIGHT);
                send_json_response(true, "roamerx_mode");
            }
            else if (line == "balance_stand")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_BALANCE_STAND_MODE);
                send_json_response(true, "balance_stand");
            }
            else if (line == "enter_entertainment")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_ENTER_ENTERTAINMENT_MODE);
                send_json_response(true, "enter_entertainment");
            }
            else if (line == "exit_entertainment")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_EXIT_ENTERTAINMENT_MODE);
                send_json_response(true, "exit_entertainment");
            }
            else if (line == "unload_squat")
            {
                zsibot_exec.SetCmd(CmdCode::CMD_UNLOAD_SQUAT);
                send_json_response(true, "unload_squat");
            }
            else if (line == "get_status")
            {
                send_json_response(true, "get_status", "", make_status_json(zsibot_exec));
            }
            else if (line == "quit")
            {
                zsibot_exec.SetRemote({0, 0, 0, 0}, std::array<float32_t, 14>{0});
                send_json_response(true, "quit");
                break;
            }
            else
            {
                send_json_response(false, line, "unknown command");
            }
        }
        catch (const std::exception &e)
        {
            send_json_response(false, line, e.what());
        }
    }
}

int interactive_mode(const RobotConfig &config)
{
    set_conio_terminal_mode();

    ZsibotExecutor zsibot_exec(Role::ROLE_SDK, config.dog_ip, config.send_port, config.recv_port);

    std::bitset<256> pressed_keys;

    while (true)
    {
        if (kbhit())
        {
            char ch = getchar();

            pressed_keys.set(static_cast<unsigned char>(ch), true);

            switch (ch)
            {
                case 'w':
                    zsibot_exec.SetRemote({0.1f, 0, 0, 0}, std::array<float32_t, 14>{0});
                    std::cout << "向前移动" << std::endl;
                    break;
                case 's':
                    zsibot_exec.SetRemote({-0.1f, 0, 0, 0}, std::array<float32_t, 14>{0});
                    std::cout << "向后移动" << std::endl;
                    break;
                case 'a':
                    zsibot_exec.SetRemote({0, 0, 0.04f, 0}, std::array<float32_t, 14>{0});
                    std::cout << "向左转" << std::endl;
                    break;
                case 'd':
                    zsibot_exec.SetRemote({0, 0, -0.05f, 0}, std::array<float32_t, 14>{0});
                    std::cout << "向右转" << std::endl;
                    break;
                case 'q':
                    zsibot_exec.SetRemote({0, 0.04f, 0, 0}, std::array<float32_t, 14>{0});
                    std::cout << "左移动" << std::endl;
                    break;
                case 'e':
                    zsibot_exec.SetRemote({0, -0.02f, 0, 0}, std::array<float32_t, 14>{0});
                    std::cout << "右移动" << std::endl;
                    break;
                case 'c':
                    zsibot_exec.SetRemote({0, 0, 0, 0}, std::array<float32_t, 14>{0});
                    std::cout << "停止移动" << std::endl;
                    break;
                case '0':
                    zsibot_exec.SetCmd(CmdCode::CMD_EMERGENCY_STOP);
                    std::cout << "切换到失能状态，机器狗软急停,完全爬下" << std::endl;
                    break;
                case '1':
                    zsibot_exec.SetCmd(CmdCode::CMD_SIT_DOWN);
                    std::cout << "切换到匍匐" << std::endl;
                    break;
                case '2':
                    zsibot_exec.SetCmd(CmdCode::CMD_STAND_UP);
                    std::cout << "切换到站立" << std::endl;
                    break;
                case '3':
                    zsibot_exec.SetCmd(CmdCode::CMD_JUMP);
                    std::cout << "切换到跳跃" << std::endl;
                    break;
                case '4':
                    zsibot_exec.SetCmd(CmdCode::CMD_FORWARD_JUMP);
                    std::cout << "切换到向前跳跃" << std::endl;
                    break;
                case '5':
                    zsibot_exec.SetCmd(CmdCode::CMD_GREET);
                    std::cout << "切换到打招呼" << std::endl;
                    break;
                case '6':
                    zsibot_exec.SetCmd(CmdCode::CMD_BACK_FLIP);
                    std::cout << "切换到后空翻" << std::endl;
                    break;
                case '7':
                    zsibot_exec.SetCmd(CmdCode::CMD_ENTER_LAB_MODE);
                    std::cout << "切换到实验室模式" << std::endl;
                    break;
                case '8':
                    zsibot_exec.SetCmd(CmdCode::CMD_EXIT_LAB_MODE);
                    std::cout << "切换到退出实验室模式" << std::endl;
                    break;
                case '9':
                    zsibot_exec.SetCmd(CmdCode::CMD_TWO_LEG_STAND);
                    std::cout << "切换到双腿站立" << std::endl;
                    break;
                case 'k':
                    zsibot_exec.SetCmd(CmdCode::CMD_SDK_CONTROL_RIGHT);
                    std::cout << "切换到sdk模式" << std::endl;
                    break;
                case 'r':
                    zsibot_exec.SetCmd(CmdCode::CMD_REMOTE_CONTROL_RIGHT);
                    std::cout << "切换到remote模式" << std::endl;
                    break;
                case 'x':
                    zsibot_exec.SetCmd(CmdCode::CMD_ROAMERX_CONTROL_RIGHT);
                    std::cout << "切换到roamerX模式" << std::endl;
                    break;
                case 'b':
                    zsibot_exec.SetCmd(CmdCode::CMD_BALANCE_STAND_MODE);
                    std::cout << "切换到平衡站立模式" << std::endl;
                    break;
                case 'l':
                    zsibot_exec.SetCmd(CmdCode::CMD_ENTER_ENTERTAINMENT_MODE);
                    std::cout << "切换到文娱模式" << std::endl;
                    break;
                case 'm':
                    zsibot_exec.SetCmd(CmdCode::CMD_EXIT_ENTERTAINMENT_MODE);
                    std::cout << "切换到退出文娱模式" << std::endl;
                    break;
                case 'u':
                    zsibot_exec.SetCmd(CmdCode::CMD_UNLOAD_SQUAT);
                    std::cout << "切换到卸货下蹲模式" << std::endl;
                    break;
                case 'i':
                {
                    std::cout << "\n========== 设备状态信息 ==========" << std::endl;
                    std::cout << "电量: " << zsibot_exec.GetPower() << "%" << std::endl;
                    std::cout << "温度: " << zsibot_exec.GetTemperature() << "°C" << std::endl;
                    std::cout << "速度等级: " << static_cast<int>(zsibot_exec.GetSpeedLevel()) << std::endl;
                    std::cout << "功能模式: " << static_cast<int>(zsibot_exec.GetFunctionMode()) << std::endl;
                    std::cout << "控制模式: " << static_cast<int>(zsibot_exec.GetControlMode()) << std::endl;
                    std::cout << "运动模式: " << static_cast<int>(zsibot_exec.GetMotionMode()) << std::endl;
                    std::cout << "运动类型: " << static_cast<int>(zsibot_exec.GetMotionType()) << std::endl;

                    SpeedInfo speed = zsibot_exec.GetSpeed();
                    std::cout << "速度: " << speed.speed << " m/s" << std::endl;
                    std::cout << "角速度: " << speed.angle_speed << " rad/s" << std::endl;
                    std::cout << "平移速度: " << speed.shift_speed << " m/s" << std::endl;
                    std::cout << "角度: " << speed.angle << " rad" << std::endl;

                    std::array<float32_t, 16> motor_temps = zsibot_exec.GetMotorTemp();
                    std::cout << "电机温度: ";
                    for (size_t i = 0; i < motor_temps.size() && i < 4; ++i)
                    {
                        std::cout << "M" << i << ": " << motor_temps[i] << "°C ";
                    }
                    std::cout << std::endl;

                    std::vector<FaultInfo> faults = zsibot_exec.GetFaultInfo();
                    if (faults.empty())
                    {
                        std::cout << "无故障信息" << std::endl;
                    }
                    else
                    {
                        std::cout << "故障信息:" << std::endl;
                        for (const auto &fault : faults)
                        {
                            std::cout << "  模块: " << fault.module << ", 子模块: " << fault.submodule
                                      << ", 错误码: " << fault.error_code << ", 等级: " << fault.level
                                      << ", 信息: " << fault.info << std::endl;
                        }
                    }
                    std::cout << "================================\n" << std::endl;
                    break;
                }
                case 'o':
                    std::cout << "设置WiFi功能演示 (不会实际执行)" << std::endl;
                    break;
                default:
                    break;
            }
        }

        std::this_thread::sleep_for(std::chrono::milliseconds(20));
    }

    reset_terminal_mode();
    return 0;
}

int main(int argc, char *argv[])
{
    RobotConfig config;
    bool daemon = false;
    bool help_requested = false;
    if (!parse_args(argc, argv, daemon, config, help_requested))
    {
        return 1;
    }
    if (help_requested)
    {
        return 0;
    }

    if (daemon)
    {
        daemon_mode(config);
        return 0;
    }

    return interactive_mode(config);
}
