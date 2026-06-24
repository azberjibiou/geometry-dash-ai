#ifdef GEODE_IS_WINDOWS
    #ifndef WIN32_LEAN_AND_MEAN
        #define WIN32_LEAN_AND_MEAN
    #endif
    #include <winsock2.h>
    #include <ws2tcpip.h>
#endif

#include <Geode/Geode.hpp>
#include <Geode/modify/GJBaseGameLayer.hpp>
#include <Geode/modify/PlayLayer.hpp>
#include <matjson.hpp>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <deque>
#include <mutex>
#include <optional>
#include <string>
#include <string_view>
#include <thread>
#include <utility>
#include <vector>

using namespace geode::prelude;
using namespace std::chrono_literals;

namespace {

constexpr int kProtocolVersion = 1;
constexpr int kDefaultBridgePort = 29430;
constexpr int kJumpButton = static_cast<int>(PlayerButton::Jump);
constexpr std::size_t kMaxOutboundMessages = 512;

enum class BridgeCommandKind {
    Action,
    LoadMacro,
    Reset,
};

struct MacroEvent {
    int tick = 0;
    bool down = false;
    bool player2 = false;
    int index = 0;
};

struct BridgeCommand {
    BridgeCommandKind kind = BridgeCommandKind::Action;
    bool down = false;
    bool player2 = false;
    int tick = 0;
    std::string reason;
    std::vector<MacroEvent> macroEvents;
};

std::string compactJson(matjson::Value const& value) {
    return value.dump(matjson::NO_INDENTATION) + "\n";
}

std::string makeAckMessage(std::string_view message, int tick) {
    return compactJson(matjson::makeObject({
        { "version", kProtocolVersion },
        { "type", "ack" },
        { "tick", tick >= 0 ? matjson::Value(tick) : matjson::Value(nullptr) },
        { "message", std::string(message) },
    }));
}

std::string makeErrorMessage(std::string_view message) {
    return compactJson(matjson::makeObject({
        { "version", kProtocolVersion },
        { "type", "error" },
        { "message", std::string(message) },
    }));
}

std::string makeDiagnosticMessage(std::string_view kind, int tick, matjson::Value data) {
    return compactJson(matjson::makeObject({
        { "version", kProtocolVersion },
        { "type", "diagnostic" },
        { "kind", std::string(kind) },
        { "tick", tick >= 0 ? matjson::Value(tick) : matjson::Value(nullptr) },
        { "data", std::move(data) },
    }));
}

template <class T>
std::optional<T> readJsonField(matjson::Value const& value, std::string_view key) {
    auto result = value.get<T>(key);
    if (result.isErr()) {
        return std::nullopt;
    }
    return std::move(result).unwrap();
}

std::optional<matjson::Value const*> readJsonObjectField(
    matjson::Value const& value,
    std::string_view key
) {
    auto result = value.get(key);
    if (result.isErr()) {
        return std::nullopt;
    }
    auto const& child = result.unwrap();
    if (!child.isObject()) {
        return std::nullopt;
    }
    return &child;
}

std::optional<matjson::Value const*> readJsonArrayField(
    matjson::Value const& value,
    std::string_view key
) {
    auto result = value.get(key);
    if (result.isErr()) {
        return std::nullopt;
    }
    auto const& child = result.unwrap();
    if (!child.isArray()) {
        return std::nullopt;
    }
    return &child;
}

void sortMacroEvents(std::vector<MacroEvent>& events) {
    std::stable_sort(events.begin(), events.end(), [](MacroEvent const& a, MacroEvent const& b) {
        if (a.tick != b.tick) {
            return a.tick < b.tick;
        }
        if (a.player2 != b.player2) {
            return !a.player2 && b.player2;
        }
        if (a.down != b.down) {
            return a.down && !b.down;
        }
        return a.index < b.index;
    });

    for (std::size_t index = 0; index < events.size(); ++index) {
        events[index].index = static_cast<int>(index);
    }
}

std::string modeFor(PlayerObject* player) {
    if (!player) {
        return "unknown";
    }
    if (player->m_isShip) {
        return "ship";
    }
    if (player->m_isBird) {
        return "ufo";
    }
    if (player->m_isBall) {
        return "ball";
    }
    if (player->m_isDart) {
        return "wave";
    }
    if (player->m_isRobot) {
        return "robot";
    }
    if (player->m_isSpider) {
        return "spider";
    }
    if (player->m_isSwing) {
        return "swing";
    }
    return "cube";
}

class BridgeServer {
public:
    BridgeServer() = default;

    BridgeServer(BridgeServer const&) = delete;
    BridgeServer& operator=(BridgeServer const&) = delete;

    ~BridgeServer() {
        this->stop();
    }

    void start(int port = kDefaultBridgePort) {
#ifdef GEODE_IS_WINDOWS
        bool expected = false;
        if (!m_started.compare_exchange_strong(expected, true)) {
            return;
        }

        m_stop.store(false);
        m_port = port;
        m_thread = std::thread([this] {
            this->run();
        });
#else
        log::warn("Geometry Dash AI bridge is currently implemented for Windows builds only");
#endif
    }

    void stop() {
#ifdef GEODE_IS_WINDOWS
        if (!m_started.exchange(false)) {
            return;
        }
        m_stop.store(true);
        if (auto socket = m_listenSocket.exchange(INVALID_SOCKET); socket != INVALID_SOCKET) {
            closesocket(socket);
        }
        m_outboundCv.notify_all();
        if (m_thread.joinable()) {
            m_thread.join();
        }
#endif
    }

    bool isClientConnected() const {
        return m_clientConnected.load();
    }

    void setCurrentTick(int tick) {
        m_currentTick.store(tick);
    }

    void enqueueObservation(std::string line) {
        if (!m_clientConnected.load()) {
            return;
        }

        {
            std::lock_guard lock(m_outboundMutex);
            if (m_outbound.size() >= kMaxOutboundMessages) {
                m_outbound.pop_front();
            }
            m_outbound.push_back(std::move(line));
        }
        m_outboundCv.notify_one();
    }

    std::vector<BridgeCommand> takeCommands() {
        std::vector<BridgeCommand> commands;
        std::lock_guard lock(m_commandMutex);
        commands.swap(m_commands);
        return commands;
    }

private:
#ifdef GEODE_IS_WINDOWS
    void run() {
        WSADATA data = {};
        if (WSAStartup(MAKEWORD(2, 2), &data) != 0) {
            log::error("Geometry Dash AI bridge failed to start Winsock");
            m_started.store(false);
            return;
        }

        SOCKET listenSocket = socket(AF_INET, SOCK_STREAM, IPPROTO_TCP);
        if (listenSocket == INVALID_SOCKET) {
            log::error("Geometry Dash AI bridge could not create socket: {}", WSAGetLastError());
            WSACleanup();
            m_started.store(false);
            return;
        }
        m_listenSocket.store(listenSocket);

        int reuse = 1;
        setsockopt(listenSocket, SOL_SOCKET, SO_REUSEADDR, reinterpret_cast<char*>(&reuse), sizeof(reuse));

        sockaddr_in address = {};
        address.sin_family = AF_INET;
        address.sin_port = htons(static_cast<u_short>(m_port));
        address.sin_addr.s_addr = htonl(INADDR_LOOPBACK);

        if (bind(listenSocket, reinterpret_cast<sockaddr*>(&address), sizeof(address)) == SOCKET_ERROR) {
            log::error(
                "Geometry Dash AI bridge could not bind 127.0.0.1:{}: {}",
                m_port,
                WSAGetLastError()
            );
            closesocket(listenSocket);
            m_listenSocket.store(INVALID_SOCKET);
            WSACleanup();
            m_started.store(false);
            return;
        }

        if (listen(listenSocket, 1) == SOCKET_ERROR) {
            log::error("Geometry Dash AI bridge listen failed: {}", WSAGetLastError());
            closesocket(listenSocket);
            m_listenSocket.store(INVALID_SOCKET);
            WSACleanup();
            m_started.store(false);
            return;
        }

        u_long nonblocking = 1;
        ioctlsocket(listenSocket, FIONBIO, &nonblocking);
        log::info("Geometry Dash AI bridge listening on 127.0.0.1:{}", m_port);

        while (!m_stop.load()) {
            fd_set readSet;
            FD_ZERO(&readSet);
            FD_SET(listenSocket, &readSet);
            timeval timeout = {};
            timeout.tv_sec = 0;
            timeout.tv_usec = 200000;

            auto ready = select(0, &readSet, nullptr, nullptr, &timeout);
            if (ready == SOCKET_ERROR) {
                if (m_stop.load()) {
                    break;
                }
                log::warn("Geometry Dash AI bridge select failed: {}", WSAGetLastError());
                continue;
            }
            if (ready <= 0 || !FD_ISSET(listenSocket, &readSet)) {
                continue;
            }

            SOCKET clientSocket = accept(listenSocket, nullptr, nullptr);
            if (clientSocket == INVALID_SOCKET) {
                if (!m_stop.load()) {
                    log::warn("Geometry Dash AI bridge accept failed: {}", WSAGetLastError());
                }
                continue;
            }

            u_long clientNonblocking = 1;
            ioctlsocket(clientSocket, FIONBIO, &clientNonblocking);
            m_clientConnected.store(true);
            log::info("Geometry Dash AI bridge client connected");
            this->serveClient(clientSocket);
            m_clientConnected.store(false);
            this->clearOutbound();
            closesocket(clientSocket);
            log::info("Geometry Dash AI bridge client disconnected");
        }

        if (auto socket = m_listenSocket.exchange(INVALID_SOCKET); socket != INVALID_SOCKET) {
            closesocket(socket);
        }
        WSACleanup();
    }

    void serveClient(SOCKET clientSocket) {
        std::string inbound;

        while (!m_stop.load()) {
            bool didWork = false;

            fd_set readSet;
            FD_ZERO(&readSet);
            FD_SET(clientSocket, &readSet);
            timeval timeout = {};
            timeout.tv_sec = 0;
            timeout.tv_usec = 10000;

            auto ready = select(0, &readSet, nullptr, nullptr, &timeout);
            if (ready == SOCKET_ERROR) {
                return;
            }
            if (ready > 0 && FD_ISSET(clientSocket, &readSet)) {
                char buffer[4096];
                int received = recv(clientSocket, buffer, sizeof(buffer), 0);
                if (received == 0) {
                    return;
                }
                if (received == SOCKET_ERROR) {
                    int error = WSAGetLastError();
                    if (error != WSAEWOULDBLOCK) {
                        return;
                    }
                }
                else {
                    didWork = true;
                    inbound.append(buffer, static_cast<std::size_t>(received));
                    if (!this->processInbound(clientSocket, inbound)) {
                        return;
                    }
                }
            }

            auto outbound = this->takeOutboundBatch();
            for (auto const& line : outbound) {
                didWork = true;
                if (!this->sendAll(clientSocket, line)) {
                    return;
                }
            }

            if (!didWork) {
                std::unique_lock lock(m_outboundMutex);
                m_outboundCv.wait_for(lock, 5ms, [this] {
                    return m_stop.load() || !m_outbound.empty();
                });
            }
        }
    }

    bool processInbound(SOCKET clientSocket, std::string& inbound) {
        while (true) {
            auto newline = inbound.find('\n');
            if (newline == std::string::npos) {
                if (inbound.size() > 65536) {
                    this->sendAll(clientSocket, makeErrorMessage("incoming message is too large"));
                    inbound.clear();
                }
                return true;
            }

            auto line = inbound.substr(0, newline);
            inbound.erase(0, newline + 1);
            if (!line.empty() && line.back() == '\r') {
                line.pop_back();
            }
            if (line.empty()) {
                continue;
            }

            auto response = this->handleProtocolLine(line);
            if (!response.empty() && !this->sendAll(clientSocket, response)) {
                return false;
            }
        }
    }

    std::string handleProtocolLine(std::string_view line) {
        auto parsed = matjson::Value::parse(line);
        if (parsed.isErr()) {
            return makeErrorMessage(std::string(parsed.unwrapErr()));
        }

        auto message = std::move(parsed).unwrap();
        if (!message.isObject()) {
            return makeErrorMessage("message must be a JSON object");
        }

        auto version = readJsonField<int>(message, "version");
        if (!version || *version != kProtocolVersion) {
            return makeErrorMessage("unsupported protocol version");
        }

        auto type = readJsonField<std::string>(message, "type");
        if (!type) {
            return makeErrorMessage("message.type is invalid");
        }

        if (*type == "action") {
            auto event = readJsonObjectField(message, "event");
            if (!event) {
                return makeErrorMessage("action message must contain event object");
            }

            auto tick = readJsonField<int>(**event, "tick");
            auto kind = readJsonField<std::string>(**event, "kind");
            auto player = readJsonField<std::string>(**event, "player").value_or("p1");
            if (!tick || !kind) {
                return makeErrorMessage("event.tick and event.kind are required");
            }
            if (*kind != "press" && *kind != "release") {
                return makeErrorMessage("event.kind must be 'press' or 'release'");
            }
            if (player != "p1" && player != "p2") {
                return makeErrorMessage("event.player must be 'p1' or 'p2'");
            }

            this->pushCommand(BridgeCommand {
                .kind = BridgeCommandKind::Action,
                .down = *kind == "press",
                .player2 = player == "p2",
                .tick = *tick,
            });
            return makeAckMessage("action queued", m_currentTick.load());
        }

        if (*type == "load_macro") {
            auto events = readJsonArrayField(message, "events");
            if (!events) {
                return makeErrorMessage("load_macro message must contain events array");
            }

            std::vector<MacroEvent> macroEvents;
            macroEvents.reserve((**events).size());
            int eventIndex = 0;
            for (auto const& eventValue : **events) {
                if (!eventValue.isObject()) {
                    return makeErrorMessage("macro event must be an object");
                }

                auto tick = readJsonField<int>(eventValue, "tick");
                auto kind = readJsonField<std::string>(eventValue, "kind");
                auto player = readJsonField<std::string>(eventValue, "player").value_or("p1");
                if (!tick || !kind) {
                    return makeErrorMessage("event.tick and event.kind are required");
                }
                if (*tick < 0) {
                    return makeErrorMessage("event.tick must be non-negative");
                }
                if (*kind != "press" && *kind != "release") {
                    return makeErrorMessage("event.kind must be 'press' or 'release'");
                }
                if (player != "p1" && player != "p2") {
                    return makeErrorMessage("event.player must be 'p1' or 'p2'");
                }

                macroEvents.push_back(MacroEvent {
                    .tick = *tick,
                    .down = *kind == "press",
                    .player2 = player == "p2",
                    .index = eventIndex++,
                });
            }
            sortMacroEvents(macroEvents);

            this->pushCommand(BridgeCommand {
                .kind = BridgeCommandKind::LoadMacro,
                .macroEvents = std::move(macroEvents),
            });
            return makeAckMessage("macro loaded", m_currentTick.load());
        }

        if (*type == "reset") {
            auto reason = readJsonField<std::string>(message, "reason").value_or("requested");
            this->pushCommand(BridgeCommand {
                .kind = BridgeCommandKind::Reset,
                .reason = reason,
            });
            return makeAckMessage("reset queued", m_currentTick.load());
        }

        return makeErrorMessage("unexpected client message");
    }

    void pushCommand(BridgeCommand command) {
        std::lock_guard lock(m_commandMutex);
        m_commands.push_back(std::move(command));
    }

    std::deque<std::string> takeOutboundBatch() {
        std::deque<std::string> batch;
        std::lock_guard lock(m_outboundMutex);
        batch.swap(m_outbound);
        return batch;
    }

    void clearOutbound() {
        std::lock_guard lock(m_outboundMutex);
        m_outbound.clear();
    }

    bool sendAll(SOCKET socket, std::string const& line) {
        char const* cursor = line.data();
        int remaining = static_cast<int>(line.size());

        while (remaining > 0 && !m_stop.load()) {
            int sent = send(socket, cursor, remaining, 0);
            if (sent == SOCKET_ERROR) {
                int error = WSAGetLastError();
                if (error == WSAEWOULDBLOCK) {
                    std::this_thread::sleep_for(1ms);
                    continue;
                }
                return false;
            }
            cursor += sent;
            remaining -= sent;
        }
        return remaining == 0;
    }
#endif

    std::atomic<bool> m_started = false;
    std::atomic<bool> m_stop = false;
    std::atomic<bool> m_clientConnected = false;
    std::atomic<int> m_currentTick = -1;
    int m_port = kDefaultBridgePort;
    std::thread m_thread;

    std::mutex m_commandMutex;
    std::vector<BridgeCommand> m_commands;

    std::mutex m_outboundMutex;
    std::condition_variable m_outboundCv;
    std::deque<std::string> m_outbound;

#ifdef GEODE_IS_WINDOWS
    std::atomic<SOCKET> m_listenSocket = INVALID_SOCKET;
#endif
};

BridgeServer& bridgeServer() {
    static BridgeServer server;
    return server;
}

struct AttemptState {
    PlayLayer* layer = nullptr;
    int nextObservationTick = 0;
    bool p1Down = false;
    bool p2Down = false;
    bool macroLoaded = false;
    bool macroActive = false;
    std::size_t nextMacroEventIndex = 0;
    std::vector<MacroEvent> loadedMacro;
};

AttemptState& attemptState() {
    static AttemptState state;
    return state;
}

void resetAttemptState(PlayLayer* layer) {
    auto& state = attemptState();
    state.layer = layer;
    state.nextObservationTick = 0;
    state.p1Down = false;
    state.p2Down = false;
    state.macroActive = state.macroLoaded;
    state.nextMacroEventIndex = 0;
    bridgeServer().setCurrentTick(0);
}

void ensureAttemptState(PlayLayer* layer) {
    auto& state = attemptState();
    if (state.layer != layer) {
        resetAttemptState(layer);
    }
}

void applyInputEvent(PlayLayer* layer, bool down, bool player2) {
    auto& state = attemptState();
    layer->handleButton(down, kJumpButton, !player2);
    if (player2) {
        state.p2Down = down;
    }
    else {
        state.p1Down = down;
    }
}

void applyBridgeCommands(PlayLayer* layer) {
    auto commands = bridgeServer().takeCommands();
    if (commands.empty()) {
        return;
    }

    ensureAttemptState(layer);
    auto& state = attemptState();

    for (auto const& command : commands) {
        if (command.kind == BridgeCommandKind::LoadMacro) {
            state.loadedMacro = command.macroEvents;
            state.macroLoaded = true;
            state.macroActive = false;
            state.nextMacroEventIndex = 0;
            continue;
        }

        if (command.kind == BridgeCommandKind::Reset) {
            state.p1Down = false;
            state.p2Down = false;
            layer->resetLevel();
            resetAttemptState(layer);
            continue;
        }

        applyInputEvent(layer, command.down, command.player2);
    }
}

void applyLoadedMacroEvents(PlayLayer* layer) {
    ensureAttemptState(layer);
    auto& state = attemptState();
    if (!state.macroActive) {
        return;
    }

    auto const attemptTick = state.nextObservationTick;
    while (
        state.nextMacroEventIndex < state.loadedMacro.size()
        && state.loadedMacro[state.nextMacroEventIndex].tick <= attemptTick
    ) {
        auto const& event = state.loadedMacro[state.nextMacroEventIndex];
        applyInputEvent(layer, event.down, event.player2);
        bridgeServer().enqueueObservation(makeDiagnosticMessage(
            "macro_event_applied",
            attemptTick,
            matjson::makeObject({
                { "event_index", event.index },
                { "intended_tick", event.tick },
                { "applied_tick", attemptTick },
                { "kind", event.down ? matjson::Value("press") : matjson::Value("release") },
                { "player", event.player2 ? matjson::Value("p2") : matjson::Value("p1") },
            })
        ));
        state.nextMacroEventIndex += 1;
    }
}

std::string makeObservationMessage(PlayLayer* layer) {
    ensureAttemptState(layer);
    auto& state = attemptState();
    auto* player = layer->m_player1;
    auto position = player ? player->getPosition() : cocos2d::CCPointZero;
    auto percent = std::clamp(layer->getCurrentPercent(), 0.0f, 100.0f);
    auto tick = state.nextObservationTick++;

    auto observation = matjson::makeObject({
        { "tick", tick },
        { "x", static_cast<double>(position.x) },
        { "y", static_cast<double>(position.y) },
        { "y_vel", player ? player->m_yVelocity : 0.0 },
        { "mode", modeFor(player) },
        { "gravity", player && player->m_isUpsideDown ? "reverse" : "normal" },
        { "percent", static_cast<double>(percent) },
        { "dead", static_cast<bool>((player && player->m_isDead) || layer->m_playerDied) },
        { "input_down", state.p1Down },
        { "x_vel", player ? static_cast<double>(player->m_playerSpeed) : 0.0 },
        { "rotation", player ? static_cast<double>(player->getRotation()) : 0.0 },
        { "death_reason", player && player->m_isDead ? matjson::Value("player_dead") : matjson::Value(nullptr) },
    });

    bridgeServer().setCurrentTick(tick);
    return compactJson(matjson::makeObject({
        { "version", kProtocolVersion },
        { "type", "observation" },
        { "observation", observation },
    }));
}

} // namespace

class $modify(AIBridgeGameLayer, GJBaseGameLayer) {
    void update(float dt) {
        auto* playLayer = typeinfo_cast<PlayLayer*>(this);
        if (playLayer && bridgeServer().isClientConnected()) {
            applyBridgeCommands(playLayer);
            applyLoadedMacroEvents(playLayer);
        }

        GJBaseGameLayer::update(dt);

        if (playLayer && bridgeServer().isClientConnected()) {
            bridgeServer().enqueueObservation(makeObservationMessage(playLayer));
        }
    }
};

class $modify(AIBridgePlayLayer, PlayLayer) {
    bool init(GJGameLevel* level, bool useReplay, bool dontCreateObjects) {
        if (!PlayLayer::init(level, useReplay, dontCreateObjects)) {
            return false;
        }
        resetAttemptState(this);
        return true;
    }

    void resetLevel() {
        PlayLayer::resetLevel();
        resetAttemptState(this);
    }

    void resetLevelFromStart() {
        PlayLayer::resetLevelFromStart();
        resetAttemptState(this);
    }

    void fullReset() {
        PlayLayer::fullReset();
        resetAttemptState(this);
    }
};

$on_mod(Loaded) {
    log::info("Geometry Dash AI Bridge loaded");
    bridgeServer().start(kDefaultBridgePort);
}
