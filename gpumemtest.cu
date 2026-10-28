// =====================================================================================
//  gpumemtest.cu - стресс-тест видеопамяти NVIDIA GPU с обнаружением ошибок в данных.
//  Журнал пишется одновременно в консоль и в файл.
//
//  Целевая карта: GeForce RTX 4090 (Ada Lovelace, sm_89, 24 ГБ GDDR6X, шина 384 бит).
//  Подходит и для других GPU - достаточно собрать под нужную архитектуру (-arch).
//
//  Как устроено
//  ------------
//  * Почти вся свободная VRAM выделяется блоками (chunks, по умолчанию 256 МБ), но не
//    больше бюджета видеопамяти, который Windows (WDDM) даёт процессу: иначе часть блоков
//    вытесняется в ОЗУ. Бюджет проверяется и во время теста. Скорость записи каждого блока
//    замеряется; блоки, размещённые драйвером в системной памяти ("sysmem fallback"),
//    исключаются из теста.
//  * Тесты (addr, random, walk1, walk0, movinv, mod20, copy, stress, fade) крутятся
//    по кругу. Ожидаемое значение каждого 64-битного слова вычисляется на лету по его
//    индексу или адресу, поэтому эталонная копия данных не нужна и проверяется вся
//    выделенная память.
//  * Ядро проверки при несовпадении атомарно добавляет запись (адрес, ожидалось,
//    прочитано) в буфер ошибок. После каждой фазы хост забирает записи, повторно
//    читает сбойные слова (ошибка хранения или чтения), считает статистику по битам
//    и адресам и пишет журнал.
//  * Один запуск ядра обрабатывает один блок за доли миллисекунды, поэтому сторожевой
//    таймер Windows (TDR) не срабатывает даже на GPU, к которому подключён монитор.
//
//  Сборка:  build.bat                                                          (Windows)
//           nvcc -O3 -std=c++17 -arch=sm_89 -o gpumemtest gpumemtest.cu -ldl   (Linux)
// =====================================================================================

#ifndef _CRT_SECURE_NO_WARNINGS
#define _CRT_SECURE_NO_WARNINGS
#endif
#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#include <io.h>
#include <dxgi1_4.h>
#pragma comment(lib, "dxgi.lib")
#else
#include <dlfcn.h>
#include <unistd.h>
#endif

#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdarg>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <map>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

typedef unsigned long long u64;

#define GMT_VERSION "1.0"
#define GMT_STR2(...) #__VA_ARGS__
#define GMT_STR(...) GMT_STR2(__VA_ARGS__)
#ifdef __CUDA_ARCH_LIST__
#define GMT_ARCHS GMT_STR(__CUDA_ARCH_LIST__)
#else
#define GMT_ARCHS "?"
#endif

// =====================================================================================
//  Остановка: Ctrl+C, закрытие окна консоли, лимит времени
// =====================================================================================
static std::atomic<bool> g_stop{false};
static std::atomic<bool> g_closing{false};  // окно консоли закрывается - не ждём Enter
static std::atomic<bool> g_done{false};     // итоговый отчёт записан

#ifdef _WIN32
static BOOL WINAPI on_console_ctrl(DWORD type) {
    switch (type) {
    case CTRL_C_EVENT:
    case CTRL_BREAK_EVENT:
        // первое нажатие - мягкая остановка с отчётом, второе - стандартное завершение
        return g_stop.exchange(true) ? FALSE : TRUE;
    case CTRL_CLOSE_EVENT:
    case CTRL_LOGOFF_EVENT:
    case CTRL_SHUTDOWN_EVENT:
        // после возврата из обработчика Windows завершит процесс: даём основному
        // потоку до ~4.5 с, чтобы дописать итог в журнал
        g_closing = true;
        g_stop = true;
        for (int i = 0; i < 90 && !g_done; ++i) Sleep(50);
        return TRUE;
    }
    return FALSE;
}

static bool own_console() {  // запущен двойным щелчком - окно закроется сразу после выхода
    DWORD ids[4];
    return GetConsoleProcessList(ids, 4) == 1;
}
#else
static void on_signal(int) {
    if (g_stop.exchange(true)) _exit(130);
}
#endif

// =====================================================================================
//  Утилиты
// =====================================================================================
static std::string strf(const char* fmt, ...) {
    char buf[2048];
    va_list ap;
    va_start(ap, fmt);
    vsnprintf(buf, sizeof buf, fmt, ap);
    va_end(ap);
    return buf;
}

static struct tm local_tm(time_t t) {
    struct tm r {};
#ifdef _WIN32
    localtime_s(&r, &t);
#else
    localtime_r(&t, &r);
#endif
    return r;
}

static std::string timestamp() {  // 2026-10-28 12:34:56.789
    using namespace std::chrono;
    const auto now = system_clock::now();
    const struct tm t = local_tm(system_clock::to_time_t(now));
    const int ms = (int)(duration_cast<milliseconds>(now.time_since_epoch()).count() % 1000);
    return strf("%04d-%02d-%02d %02d:%02d:%02d.%03d", t.tm_year + 1900, t.tm_mon + 1, t.tm_mday,
                t.tm_hour, t.tm_min, t.tm_sec, ms);
}

static std::string file_stamp() {
    const struct tm t = local_tm(time(nullptr));
    char b[32];
    strftime(b, sizeof b, "%Y%m%d_%H%M%S", &t);
    return b;
}

static std::string fmt_size(double b) {  // двоичные единицы - объёмы памяти
    static const char* u[] = {"B", "KiB", "MiB", "GiB", "TiB"};
    int i = 0;
    while (b >= 1024 && i < 4) { b /= 1024; ++i; }
    return strf(i ? "%.2f %s" : "%.0f %s", b, u[i]);
}

static std::string fmt_traffic(double b) {  // десятичные единицы - объём трафика
    static const char* u[] = {"B", "KB", "MB", "GB", "TB", "PB", "EB"};
    int i = 0;
    while (b >= 1000 && i < 6) { b /= 1000; ++i; }
    return strf("%.2f %s", b, u[i]);
}

static std::string fmt_dur(double sec) {
    u64 s = (u64)(sec + 0.5);
    const u64 d = s / 86400;
    s %= 86400;
    const std::string hms = strf("%02llu:%02llu:%02llu", s / 3600, s / 60 % 60, s % 60);
    return d ? strf("%llud ", d) + hms : hms;
}

static int popc64(u64 x) { int n = 0; for (; x; x &= x - 1) ++n; return n; }
static int ctz64(u64 x) { int n = 0; while (x && !(x & 1)) { x >>= 1; ++n; } return n; }

// "[1 bit: 8; 1->0 x1, 0->1 x0]" - какие биты слова сбоят и в какую сторону
static std::string flip_str(u64 expected, u64 actual) {
    const u64 x = expected ^ actual;
    const int n = popc64(x), up = popc64(x & actual), down = n - up;
    std::string bits;
    if (n <= 8)
        for (u64 t = x; t; t &= t - 1) bits += (bits.empty() ? "" : ",") + std::to_string(ctz64(t));
    else
        bits = "...";
    return strf("[%d bit%s: %s; 1->0 x%d, 0->1 x%d]", n, n == 1 ? "" : "s", bits.c_str(), down, up);
}

// =====================================================================================
//  Журнал: консоль (с цветом) + файл. WARN/FAIL/ERROR сразу сбрасываются на диск,
//  чтобы запись пережила зависание драйвера или BSOD.
// =====================================================================================
// L_PASS - строки по каждому тесту (скрыты при --quiet), L_GOOD - итоговые "зелёные" строки
enum Level { L_DEBUG, L_INFO, L_PASS, L_MAIN, L_GOOD, L_WARN, L_FAIL, L_ERROR };

class Logger {
public:
    bool color = false, quiet = false, verbose = false;

    bool open(const std::string& path) {
        f_ = fopen(path.c_str(), "a");
        return f_ != nullptr;
    }

    void close() {
        std::lock_guard<std::mutex> lk(mu_);
        if (f_) { sync_locked(); fclose(f_); f_ = nullptr; }
    }

    void log(Level lv, const char* fmt, ...) {
        char msg[4096];
        va_list ap;
        va_start(ap, fmt);
        vsnprintf(msg, sizeof msg, fmt, ap);
        va_end(ap);

        static const char* tag[] = {"DEBUG", "INFO ", "PASS ", "INFO ", "PASS ", "WARN ", "FAIL ", "ERROR"};
        static const char* clr[] = {"\x1b[90m", "", "\x1b[92m", "\x1b[96m", "\x1b[92;1m", "\x1b[93m", "\x1b[91m", "\x1b[91;1m"};
        std::lock_guard<std::mutex> lk(mu_);
        const std::string ts = timestamp();
        if (f_ && (verbose || lv != L_DEBUG)) {
            fprintf(f_, "%s [%s] %s\n", ts.c_str(), tag[lv], msg);
            fflush(f_);
            if (lv >= L_WARN) sync_locked();
        }
        if (!(verbose || (quiet ? lv >= L_MAIN : lv != L_DEBUG))) return;
        const char* t = ts.c_str() + 11;  // в консоли - только время
        if (!color || !*clr[lv]) printf("%s [%s] %s\n", t, tag[lv], msg);
        else if (lv >= L_WARN) printf("%s %s[%s] %s\x1b[0m\n", t, clr[lv], tag[lv], msg);
        else printf("%s %s[%s]\x1b[0m %s\n", t, clr[lv], tag[lv], msg);
        fflush(stdout);
    }

private:
    FILE* f_ = nullptr;
    std::mutex mu_;

    void sync_locked() {
#ifdef _WIN32
        _commit(_fileno(f_));
#else
        fsync(fileno(f_));
#endif
    }
};

static Logger g_logger;
#define LOG(lv, ...) g_logger.log(lv, __VA_ARGS__)

static bool console_supports_color() {
#ifdef _WIN32
    HANDLE h = GetStdHandle(STD_OUTPUT_HANDLE);
    DWORD mode = 0;
    if (h == INVALID_HANDLE_VALUE || !GetConsoleMode(h, &mode)) return false;  // вывод перенаправлен
    return SetConsoleMode(h, mode | ENABLE_VIRTUAL_TERMINAL_PROCESSING) != 0;
#else
    return isatty(fileno(stdout)) != 0;
#endif
}

// =====================================================================================
//  Код GPU
// =====================================================================================

// Вид шаблона. Ожидаемое значение слова - функция от его индекса/адреса, поэтому
// эталонная копия данных не нужна.
enum PatKind { PK_CONST = 0, PK_WALK, PK_ADDR, PK_RAND, PK_MOD };

struct Pat {
    int      kind;
    u64      a;         // CONST: значение; WALK: исходное слово; ADDR: XOR-маска; RAND: seed; MOD: P
    u64      b;         // WALK: сдвиг; RAND: XOR-маска
    unsigned m;         // MOD: модуль - P лежит в каждом m-м слове, в остальных ~P
    unsigned k;         // MOD: остаток (позиция P)
    unsigned base_mod;  // MOD: (глобальный индекс начала блока) % m, заполняет хост
};

struct ErrHdr {            // заголовок буфера ошибок (в памяти GPU), обнуляется после каждой фазы
    u64      count;        // число несовпавших слов за фазу
    u64      bit_or;       // OR всех XOR(ожидалось, прочитано)
    unsigned bit_cnt[64];  // сколько раз сбоил каждый бит
};

struct ErrRec {            // запись об одной ошибке
    u64      idx;          // индекс 64-битного слова внутри блока
    u64      expected;
    u64      actual;
    unsigned chunk;
    unsigned pad;
};

struct ErrCtx {
    ErrHdr*  hdr;
    ErrRec*  rec;
    unsigned cap;
    unsigned chunk;
};

constexpr unsigned kMaxRecords = 4096;        // записей об ошибках на фазу
constexpr u64      kHistLimit  = 1ull << 20;  // гистограмма битов - по первым N ошибкам фазы
constexpr int      kBlock      = 256;

__host__ __device__ __forceinline__ u64 mix64(u64 z) {  // финализатор splitmix64
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ull;
    z = (z ^ (z >> 27)) * 0x94D049BB133111EBull;
    return z ^ (z >> 31);
}

__host__ __device__ __forceinline__ u64 rotl64(u64 x, unsigned s) {
    s &= 63;
    return (x << s) | (x >> ((64 - s) & 63));
}

// gidx - глобальный индекс слова в тестируемой области, lidx - индекс внутри блока
template <int K>
__device__ __forceinline__ u64 pat_value(const Pat& p, u64 gidx, u64 lidx, const u64* addr) {
    if constexpr (K == PK_CONST) {
        return p.a;
    } else if constexpr (K == PK_WALK) {
        return rotl64(p.a, (unsigned)(gidx + p.b));  // соседние слова - соседние позиции бита
    } else if constexpr (K == PK_ADDR) {
        return (u64)(uintptr_t)addr ^ p.a;
    } else if constexpr (K == PK_RAND) {
        return mix64(p.a + gidx * 0x9E3779B97F4A7C15ull) ^ p.b;
    } else {
        unsigned r = (unsigned)lidx % p.m + p.base_mod;
        if (r >= p.m) r -= p.m;
        return r == p.k ? p.a : ~p.a;
    }
}

// Холодный путь: вызывается только при несовпадении
__device__ __noinline__ void report_error(ErrHdr* hdr, ErrRec* rec, unsigned cap, unsigned chunk,
                                          u64 idx, u64 expected, u64 actual) {
    const u64 n = atomicAdd(&hdr->count, 1ull);
    const u64 x = expected ^ actual;
    atomicOr(&hdr->bit_or, x);
    if (n < kHistLimit)
        for (u64 t = x; t; t &= t - 1) atomicAdd(&hdr->bit_cnt[__ffsll((long long)t) - 1], 1u);
    if (n < cap) {
        ErrRec r;
        r.idx = idx;
        r.expected = expected;
        r.actual = actual;
        r.chunk = chunk;
        r.pad = 0;
        rec[n] = r;
    }
}

#define GRID_STRIDE(i, n) \
    for (u64 i = blockIdx.x * (u64)blockDim.x + threadIdx.x; i < (n); i += (u64)gridDim.x * blockDim.x)

// Доступ векторами по 16 байт (2 слова); ld/st.cg - кэшировать только в L2.
template <int K>
__global__ void k_fill(u64* buf, u64 nvec, u64 gbase, Pat p) {
    ulonglong2* v = reinterpret_cast<ulonglong2*>(buf);
    GRID_STRIDE(i, nvec) {
        const u64 l = 2 * i;
        __stcg(v + i, make_ulonglong2(pat_value<K>(p, gbase + l, l, buf + l),
                                      pat_value<K>(p, gbase + l + 1, l + 1, buf + l + 1)));
    }
}

template <int K>
__global__ void k_verify(const u64* buf, u64 nvec, u64 gbase, Pat p, ErrCtx ec) {
    const ulonglong2* v = reinterpret_cast<const ulonglong2*>(buf);
    GRID_STRIDE(i, nvec) {
        const u64 l = 2 * i;
        const ulonglong2 w = __ldcg(v + i);
        const u64 e0 = pat_value<K>(p, gbase + l, l, buf + l);
        const u64 e1 = pat_value<K>(p, gbase + l + 1, l + 1, buf + l + 1);
        if (w.x != e0) report_error(ec.hdr, ec.rec, ec.cap, ec.chunk, l, e0, w.x);
        if (w.y != e1) report_error(ec.hdr, ec.rec, ec.cap, ec.chunk, l + 1, e1, w.y);
    }
}

// Прочитать, проверить и записать (ожидаемое ^ wmask) - элемент moving inversions.
// Пишется значение от ожидаемого, а не от прочитанного, чтобы ошибка не тянулась дальше.
template <int K>
__global__ void k_verify_write(u64* buf, u64 nvec, u64 gbase, Pat p, u64 wmask, ErrCtx ec) {
    ulonglong2* v = reinterpret_cast<ulonglong2*>(buf);
    GRID_STRIDE(i, nvec) {
        const u64 l = 2 * i;
        const ulonglong2 w = __ldcg(v + i);
        const u64 e0 = pat_value<K>(p, gbase + l, l, buf + l);
        const u64 e1 = pat_value<K>(p, gbase + l + 1, l + 1, buf + l + 1);
        if (w.x != e0) report_error(ec.hdr, ec.rec, ec.cap, ec.chunk, l, e0, w.x);
        if (w.y != e1) report_error(ec.hdr, ec.rec, ec.cap, ec.chunk, l + 1, e1, w.y);
        __stcg(v + i, make_ulonglong2(e0 ^ wmask, e1 ^ wmask));
    }
}

// modulo-X: перезаписать ~P во всех словах, кроме позиций P
__global__ void k_mod_others(u64* buf, u64 nwords, Pat p) {
    const u64 inv = ~p.a;
    GRID_STRIDE(i, nwords) {
        unsigned r = (unsigned)i % p.m + p.base_mod;
        if (r >= p.m) r -= p.m;
        if (r != p.k) __stcg(buf + i, inv);
    }
}

#define DISPATCH(kind, KERNEL, ...)                                               \
    do {                                                                          \
        switch (kind) {                                                           \
        case PK_CONST: KERNEL<PK_CONST><<<grid_, kBlock>>>(__VA_ARGS__); break;   \
        case PK_WALK:  KERNEL<PK_WALK><<<grid_, kBlock>>>(__VA_ARGS__); break;    \
        case PK_ADDR:  KERNEL<PK_ADDR><<<grid_, kBlock>>>(__VA_ARGS__); break;    \
        case PK_RAND:  KERNEL<PK_RAND><<<grid_, kBlock>>>(__VA_ARGS__); break;    \
        default:       KERNEL<PK_MOD><<<grid_, kBlock>>>(__VA_ARGS__); break;     \
        }                                                                         \
    } while (0)

static Pat P_const(u64 v)               { Pat p{}; p.kind = PK_CONST; p.a = v; return p; }
static Pat P_walk(u64 base, u64 shift)  { Pat p{}; p.kind = PK_WALK; p.a = base; p.b = shift; return p; }
static Pat P_addr(u64 mask)             { Pat p{}; p.kind = PK_ADDR; p.a = mask; return p; }
static Pat P_rand(u64 seed, u64 mask)   { Pat p{}; p.kind = PK_RAND; p.a = seed; p.b = mask; return p; }
static Pat P_mod(u64 v, unsigned m, unsigned k) { Pat p{}; p.kind = PK_MOD; p.a = v; p.m = m; p.k = k; return p; }

// =====================================================================================
//  Телеметрия через NVML (nvml.dll / libnvidia-ml.so грузится динамически, необязательна)
// =====================================================================================
class Nvml {
private:
    struct TempV1 { unsigned version; int sensor; int temperature; };  // nvmlTemperature_v1_t

public:
    struct Sample { bool ok = false; int temp = -1, sm = -1, mem = -1, fan = -1; double power = -1; u64 reasons = 0; };
    std::string driver;

    bool init(const char* pci_bus_id) {
#ifdef _WIN32
        HMODULE h = LoadLibraryExA("nvml.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
        if (!h) h = LoadLibraryA("C:\\Program Files\\NVIDIA Corporation\\NVSMI\\nvml.dll");
        lib_ = (void*)h;
#else
        lib_ = dlopen("libnvidia-ml.so.1", RTLD_NOW);
#endif
        if (!lib_) return false;
        auto init_fn = (int (*)())sym("nvmlInit_v2");
        auto by_pci = (int (*)(const char*, void**))sym("nvmlDeviceGetHandleByPciBusId_v2");
        shutdown_ = (int (*)())sym("nvmlShutdown");
        if (!init_fn || !by_pci || !shutdown_ || init_fn() != 0) return false;
        if (by_pci(pci_bus_id, &dev_) != 0) { shutdown_(); return false; }
        temp_ = (int (*)(void*, int, unsigned*))sym("nvmlDeviceGetTemperature");
        tempv_ = (int (*)(void*, TempV1*))sym("nvmlDeviceGetTemperatureV");
        power_ = (int (*)(void*, unsigned*))sym("nvmlDeviceGetPowerUsage");
        fan_ = (int (*)(void*, unsigned*))sym("nvmlDeviceGetFanSpeed");
        clock_ = (int (*)(void*, int, unsigned*))sym("nvmlDeviceGetClockInfo");
        reasons_ = (int (*)(void*, unsigned long long*))sym("nvmlDeviceGetCurrentClocksEventReasons");
        if (!reasons_) reasons_ = (int (*)(void*, unsigned long long*))sym("nvmlDeviceGetCurrentClocksThrottleReasons");
        auto drv = (int (*)(char*, unsigned))sym("nvmlSystemGetDriverVersion");
        char buf[96] = {0};
        if (drv && drv(buf, sizeof buf) == 0) driver = buf;
        ok_ = true;
        return true;
    }

    Sample sample() {
        Sample s;
        if (!ok_) return s;
        s.ok = true;
        unsigned v = 0;
        if (temp_ && temp_(dev_, 0 /*NVML_TEMPERATURE_GPU*/, &v) == 0) {
            s.temp = (int)v;
        } else if (tempv_) {
            TempV1 t = {(unsigned)sizeof(TempV1) | (1u << 24), 0, 0};
            if (tempv_(dev_, &t) == 0) s.temp = t.temperature;
        }
        if (power_ && power_(dev_, &v) == 0) s.power = v / 1000.0;
        if (fan_ && fan_(dev_, &v) == 0) s.fan = (int)v;
        if (clock_ && clock_(dev_, 1 /*NVML_CLOCK_SM*/, &v) == 0) s.sm = (int)v;
        if (clock_ && clock_(dev_, 2 /*NVML_CLOCK_MEM*/, &v) == 0) s.mem = (int)v;
        unsigned long long r = 0;
        if (reasons_ && reasons_(dev_, &r) == 0) s.reasons = r;
        return s;
    }

    void shutdown() {
        if (ok_ && shutdown_) shutdown_();
        ok_ = false;
    }

private:
    void* lib_ = nullptr;
    void* dev_ = nullptr;
    bool ok_ = false;
    int (*shutdown_)() = nullptr;
    int (*temp_)(void*, int, unsigned*) = nullptr;
    int (*tempv_)(void*, TempV1*) = nullptr;
    int (*power_)(void*, unsigned*) = nullptr;
    int (*fan_)(void*, unsigned*) = nullptr;
    int (*clock_)(void*, int, unsigned*) = nullptr;
    int (*reasons_)(void*, unsigned long long*) = nullptr;

    void* sym(const char* name) {
#ifdef _WIN32
        return (void*)GetProcAddress((HMODULE)lib_, name);
#else
        return dlsym(lib_, name);
#endif
    }
};

// Причины ограничения частот (nvmlClocksEventReason*). Не показываем "GPU idle" и
// "reliability" (0x400) - штатное ограничение буста по напряжению, "VRel" в GPU-Z.
static std::string limits_str(u64 r) {
    static const struct { u64 bit; const char* name; } k[] = {
        {0x4, "power"}, {0x20, "thermal"}, {0x40, "hw-thermal"}, {0x8, "hw-slowdown"},
        {0x80, "power-brake"}, {0x200, "board"}};
    std::string s;
    for (const auto& e : k)
        if (r & e.bit) { if (!s.empty()) s += ','; s += e.name; }
    return s;
}

static std::string tele_str(const Nvml::Sample& s) {
    if (!s.ok) return "";
    std::string r;
    if (s.temp >= 0) r += strf("%dC ", s.temp);
    if (s.power >= 0) r += strf("%.0fW ", s.power);
    if (s.sm >= 0) r += strf("SM %d ", s.sm);
    if (s.mem >= 0) r += strf("MEM %d MHz ", s.mem);
    if (s.fan >= 0) r += strf("fan %d%% ", s.fan);
    const std::string lim = limits_str(s.reasons);
    if (!lim.empty()) r += "limit:" + lim + " ";
    if (!r.empty()) r.pop_back();
    return r;
}

// =====================================================================================
//  Бюджет видеопамяти процесса в WDDM (Windows, DXGI). Если процесс его превышает,
//  менеджер видеопамяти Windows вытесняет часть блоков в ОЗУ и подкачивает обратно:
//  отдельные блоки работают на скорости PCIe (10-90 ГБ/с вместо ~1000), и данные ходят
//  через системную память вместо VRAM. Бюджет меняется, когда память нужна другим
//  процессам (рабочий стол, браузер), поэтому его надо проверять и во время теста.
// =====================================================================================
#ifdef _WIN32
class WddmBudget {
public:
    bool init(const char* luid) {
        IDXGIFactory1* f = nullptr;
        if (FAILED(CreateDXGIFactory1(__uuidof(IDXGIFactory1), (void**)&f))) return false;
        IDXGIAdapter1* a = nullptr;
        for (UINT i = 0; !a3_ && f->EnumAdapters1(i, &a) != DXGI_ERROR_NOT_FOUND; ++i) {
            DXGI_ADAPTER_DESC1 d;
            if (SUCCEEDED(a->GetDesc1(&d)) && memcmp(&d.AdapterLuid, luid, sizeof d.AdapterLuid) == 0)
                a->QueryInterface(__uuidof(IDXGIAdapter3), (void**)&a3_);
            a->Release();
        }
        f->Release();
        return a3_ != nullptr;
    }
    // budget - сколько процессу можно занимать сейчас, usage - сколько занято
    bool query(u64& budget, u64& usage) const {
        DXGI_QUERY_VIDEO_MEMORY_INFO mi;
        if (!a3_ || FAILED(a3_->QueryVideoMemoryInfo(0, DXGI_MEMORY_SEGMENT_GROUP_LOCAL, &mi))) return false;
        budget = mi.Budget;
        usage = mi.CurrentUsage;
        return true;
    }
    ~WddmBudget() { if (a3_) a3_->Release(); }

private:
    IDXGIAdapter3* a3_ = nullptr;
};
#else
class WddmBudget {  // в Linux/TCC видеопамять процесса не вытесняется
public:
    bool init(const char*) { return false; }
    bool query(u64&, u64&) const { return false; }
};
#endif

// =====================================================================================
//  Описание тестов и параметры командной строки
// =====================================================================================
enum TestId { T_ADDR, T_RANDOM, T_WALK1, T_WALK0, T_MOVINV, T_MOD20, T_COPY, T_STRESS, T_FADE, T_COUNT };

struct TestDef {
    const char* name;
    const char* title;
    bool        on_by_default;
    bool        has_bw;  // имеет смысл считать пропускную способность
};

static const TestDef kTests[T_COUNT] = {
    {"addr",   "own address: every word holds its own address, then the inverse", true,  true},
    {"random", "random data: write/verify, read-verify-invert, new seed each pass", true,  true},
    {"walk1",  "walking ones: a single 1 bit through all 64 bit positions",        true,  true},
    {"walk0",  "walking zeros: a single 0 bit through all 64 bit positions",       true,  true},
    {"movinv", "moving inversions with 8 patterns (0/1, checkerboards, random)",   true,  true},
    {"mod20",  "modulo-20: P in every 20th word, ~P rewritten around it",          true,  true},
    {"copy",   "device-to-device copies of random data with verification",        true,  true},
    {"stress", "sustained read-verify-invert sweeps of random data (heats VRAM)",  true,  true},
    {"fade",   "bit fade: write 0s/1s, leave the GPU idle, verify (off by default)", false, false},
};

struct Options {
    int         device = 0;
    u64         mem_mb = 0;      // 0 - вся свободная память минус reserve
    double      mem_pct = 0;     // либо процент свободной памяти
    u64         reserve_mb = 0;  // 0 - авто: 1024 МБ в WDDM, 256 МБ иначе
    u64         chunk_mb = 256;
    double      time_sec = 0;    // 0 - без ограничения
    u64         loops = 0;       // 0 - без ограничения
    std::string tests_spec = "default";
    std::vector<int> tests;
    std::string log_path;
    int         stress_iters = 64;
    int         rand_passes = 4;
    int         fade_sec = 60;
    int         max_report = 32;
    int         inject = 0;
    bool        stop_on_error = false;
    u64         seed = 0;
    bool        seed_set = false;
    bool        quiet = false, verbose = false, color = true, pause = true, list = false;
    std::string cmdline;
};

static void usage() {
    printf(
        "gpumemtest " GMT_VERSION " - GPU video memory stress test with data error detection\n"
        "(built for: sm %s)\n\n"
        "Usage: gpumemtest [options]\n\n"
        "  -d, --device N        CUDA device index (default 0), see --list\n"
        "  -m, --mem SIZE        VRAM to test: MiB (e.g. 20000) or percent of free memory (e.g. 90%%)\n"
        "                        default: all free memory minus --reserve\n"
        "  -r, --reserve MiB     VRAM left free for Windows/display/other apps\n"
        "                        (default 1024 on Windows WDDM, 256 otherwise)\n"
        "  -c, --chunk MiB       allocation block size, 16..4096 (default 256)\n"
        "  -t, --time DUR        stop after DUR: 90s, 30m, 2h, 1d (plain number = minutes)\n"
        "  -n, --loops N         stop after N full loops over all selected tests\n"
        "                        (without -t and -n the test runs until Ctrl+C)\n"
        "  -T, --tests LIST      comma separated: addr,random,walk1,walk0,movinv,mod20,copy,stress,fade\n"
        "                        also 'default' (all but fade) and 'all'; '-name' excludes a test,\n"
        "                        e.g. -T default,-mod20  or  -T all  or  -T random,stress\n"
        "  -l, --log FILE        log file (default gpumemtest_YYYYMMDD_HHMMSS.log, appended)\n"
        "      --stress-iters N  read-verify-invert sweeps per 'stress' run (default 64)\n"
        "      --rand-passes N   seeds per 'random' run (default 4)\n"
        "      --fade-sec N      idle time of the 'fade' test, seconds (default 60)\n"
        "      --max-report N    detailed error lines per phase (default 32)\n"
        "      --stop-on-error   stop at the first detected error\n"
        "      --inject N        SELF-TEST: flip N random bits before the first check of every test\n"
        "                        and verify that all of them are detected\n"
        "      --seed N          base seed of random patterns (default: random, written to the log)\n"
        "  -q, --quiet           console: only loop summaries, warnings and errors\n"
        "  -v, --verbose         log every test phase\n"
        "      --no-color        no ANSI colors in the console\n"
        "      --no-pause        do not wait for Enter on exit when started by double click\n"
        "      --list            list CUDA devices and exit\n"
        "  -h, --help            this help\n\n"
        "Exit code: 0 - no errors, 1 - memory errors found, 2 - aborted (CUDA/driver error), 3 - bad arguments\n",
        GMT_ARCHS);
}

static bool parse_duration(const std::string& s, double& sec) {
    char* end = nullptr;
    const double v = strtod(s.c_str(), &end);
    if (end == s.c_str() || v < 0) return false;
    const std::string u(end);
    if (u.empty() || u == "m" || u == "min") sec = v * 60;
    else if (u == "s" || u == "sec") sec = v;
    else if (u == "h") sec = v * 3600;
    else if (u == "d") sec = v * 86400;
    else return false;
    return true;
}

static bool select_tests(const std::string& spec, std::vector<int>& out, std::string& err) {
    bool on[T_COUNT] = {};
    size_t pos = 0;
    while (pos <= spec.size()) {
        size_t e = spec.find(',', pos);
        if (e == std::string::npos) e = spec.size();
        std::string tok = spec.substr(pos, e - pos);
        pos = e + 1;
        tok.erase(0, tok.find_first_not_of(" \t"));
        tok.erase(tok.find_last_not_of(" \t") + 1);
        if (tok.empty()) continue;
        bool neg = false;
        if (tok[0] == '-' || tok[0] == '+') { neg = tok[0] == '-'; tok.erase(0, 1); }
        bool known = tok == "all" || tok == "default";
        for (int i = 0; i < T_COUNT; ++i)
            if (tok == "all" || (tok == "default" && kTests[i].on_by_default) || tok == kTests[i].name) {
                on[i] = !neg;
                known = true;
            }
        if (!known) { err = "unknown test '" + tok + "'"; return false; }
    }
    out.clear();
    for (int i = 0; i < T_COUNT; ++i)
        if (on[i]) out.push_back(i);
    if (out.empty()) { err = "no tests selected"; return false; }
    return true;
}

// Возвращает -1 - продолжать, иначе код выхода. Ошибки - исключением std::invalid_argument.
static int parse_args(int argc, char** argv, Options& o) {
    for (int i = 0; i < argc; ++i) o.cmdline += (i ? " " : "") + std::string(argv[i]);
    for (int i = 1; i < argc; ++i) {
        std::string a = argv[i], inline_val;
        bool has_inline = false;
        if (a.size() > 2 && a[0] == '-' && a[1] == '-') {
            const size_t eq = a.find('=');
            if (eq != std::string::npos) { inline_val = a.substr(eq + 1); a.resize(eq); has_inline = true; }
        }
        auto val = [&]() -> std::string {
            if (has_inline) return inline_val;
            if (i + 1 >= argc) throw std::invalid_argument("option " + a + " needs a value");
            return argv[++i];
        };
        auto num = [&](u64 lo, u64 hi) -> u64 {
            const std::string s = val();
            char* end = nullptr;
            const u64 r = strtoull(s.c_str(), &end, 0);
            if (s.empty() || *end || s[0] == '-' || r < lo || r > hi)
                throw std::invalid_argument(strf("bad value for %s: '%s' (allowed %llu..%llu)", a.c_str(), s.c_str(), lo, hi));
            return r;
        };
        if (a == "-h" || a == "--help") { usage(); return 0; }
        else if (a == "-d" || a == "--device") o.device = (int)num(0, 63);
        else if (a == "-m" || a == "--mem") {
            const std::string s = val();
            char* end = nullptr;
            if (!s.empty() && s.back() == '%') {
                o.mem_pct = strtod(s.c_str(), &end);
                if (*end != '%' || o.mem_pct <= 0 || o.mem_pct > 100) throw std::invalid_argument("bad --mem percent: " + s);
            } else {
                o.mem_mb = strtoull(s.c_str(), &end, 10);
                if (s.empty() || *end || o.mem_mb == 0) throw std::invalid_argument("bad --mem value: " + s);
            }
        }
        else if (a == "-r" || a == "--reserve") o.reserve_mb = num(1, 1u << 20);
        else if (a == "-c" || a == "--chunk") o.chunk_mb = num(16, 4096);
        else if (a == "-t" || a == "--time" || a == "--duration") {
            const std::string s = val();
            if (!parse_duration(s, o.time_sec)) throw std::invalid_argument("bad duration: " + s + " (use e.g. 90s, 30m, 2h)");
        }
        else if (a == "-n" || a == "--loops") o.loops = num(1, ~0ull);
        else if (a == "-T" || a == "--tests") o.tests_spec = val();
        else if (a == "-l" || a == "--log") o.log_path = val();
        else if (a == "--stress-iters") o.stress_iters = (int)num(1, 100000);
        else if (a == "--rand-passes") o.rand_passes = (int)num(1, 10000);
        else if (a == "--fade-sec") o.fade_sec = (int)num(1, 86400);
        else if (a == "--max-report") o.max_report = (int)num(0, kMaxRecords);
        else if (a == "--inject") o.inject = (int)num(0, 1000);
        else if (a == "--stop-on-error") o.stop_on_error = true;
        else if (a == "--seed") { o.seed = num(0, ~0ull); o.seed_set = true; }
        else if (a == "-q" || a == "--quiet") o.quiet = true;
        else if (a == "-v" || a == "--verbose") o.verbose = true;
        else if (a == "--no-color") o.color = false;
        else if (a == "--no-pause") o.pause = false;
        else if (a == "--list") o.list = true;
        else throw std::invalid_argument("unknown option: " + a);
    }
    std::string err;
    if (!select_tests(o.tests_spec, o.tests, err)) throw std::invalid_argument("--tests: " + err);
    return -1;
}

static int list_devices() {
    int n = 0;
    const cudaError_t e = cudaGetDeviceCount(&n);
    if (e != cudaSuccess) { printf("CUDA error: %s\n", cudaGetErrorString(e)); return 2; }
    for (int d = 0; d < n; ++d) {
        cudaDeviceProp p;
        char pci[64] = {0};
        if (cudaGetDeviceProperties(&p, d) != cudaSuccess) continue;
        cudaDeviceGetPCIBusId(pci, sizeof pci, d);
        printf("  [%d] %s, %s, CC %d.%d, PCI %s\n", d, p.name, fmt_size((double)p.totalGlobalMem).c_str(), p.major, p.minor, pci);
    }
    return 0;
}

// =====================================================================================
//  Тестер
// =====================================================================================
struct CudaFatal {
    cudaError_t err;
    std::string where;
};
#define CK(call)                                                                                  \
    do {                                                                                          \
        const cudaError_t e_ = (call);                                                            \
        if (e_ != cudaSuccess) throw CudaFatal{e_, strf("%s, line %d", #call, __LINE__)};         \
    } while (0)

struct Chunk {
    u64* ptr;
    u64  bytes, words;
    u64  gbase;  // глобальный индекс первого слова в тестируемой области
    u64  goff;   // смещение блока в тестируемой области, байт
};

struct Inject { unsigned chunk; u64 idx; u64 orig, value; bool hit; };

struct TestStat { u64 runs = 0, errors = 0; double sec = 0, bytes = 0, base_bw = 0; bool slow = false; };

struct AddrStat { u64 count = 0, bit_or = 0, first_loop = 0, last_loop = 0, va = 0; const char* first_test = ""; };

struct DevInfo {
    std::string name, pci;
    int cc_major = 0, cc_minor = 0, sms = 0, threads_per_sm = 0;
    int bus_width = 0, mem_clock_khz = 0, l2 = 0, ecc = 0, tcc = 0, watchdog = 0;
    int drv_ver = 0, rt_ver = 0;
    u64 total = 0;
    double peak_bw = 0;  // байт/с
};

constexpr size_t kMaxAddrStats = 100000;

class Tester {
public:
    Tester(Options& o, Nvml& nv) : o_(o), nv_(nv) {}

    void init_device();
    void allocate();
    void probe();
    int  run();
    int  fatal(const CudaFatal& f);
    void release();

private:
    typedef std::chrono::steady_clock clk;
    static double since(clk::time_point t) { return std::chrono::duration<double>(clk::now() - t).count(); }
    bool should_stop() const { return g_stop || (has_deadline_ && clk::now() >= deadline_); }

    double peak_bw(int mem_mhz) const { return 2.0 * mem_mhz * 1e6 * di_.bus_width / 8; }  // DDR: 2 передачи за такт

    Pat for_chunk(const Pat& p, u64 gbase) const {
        Pat q = p;
        q.base_mod = p.m ? (unsigned)(gbase % p.m) : 0;
        return q;
    }

    void reindex();
    void fill_chunk(size_t c, const Pat& p);
    void fill(const Pat& p);
    u64  verify(const Pat& p, const std::string& what, const std::vector<u64>* gexp = nullptr);
    u64  verify_write(const Pat& p, u64 wmask, const std::string& what);
    void mod_others(const Pat& p);
    void inject(int n);
    u64  collect(const std::string& what, bool rereadable);
    std::string describe(const ErrRec& r) const;

    bool run_test(int id);
    bool t_addr();
    bool t_random();
    bool walk(u64 base, const char* name);
    bool t_movinv();
    bool t_mod20();
    bool t_copy();
    bool t_stress();
    bool t_fade();

    void check_budget();
    void note_telemetry(const Nvml::Sample& s);
    void set_title();
    std::string stop_reason() const;
    void summary(const std::string& reason, bool aborted);

    Options& o_;
    Nvml&    nv_;
    DevInfo  di_;
    std::vector<Chunk> ch_;
    u64      region_ = 0;
    u64      shrunk_ = 0;  // сколько отдали обратно из-за уменьшения бюджета WDDM
    WddmBudget wddm_;
    bool     wddm_ok_ = false;
    int      grid_ = 0;
    ErrHdr*  d_hdr_ = nullptr;
    ErrRec*  d_rec_ = nullptr;
    std::mt19937_64 rng_, inj_rng_;

    // состояние прогона
    clk::time_point t0_, deadline_;
    bool        started_ = false, has_deadline_ = false;
    u64         loop_ = 0, loops_done_ = 0;
    const char* test_ = "-";
    u64         test_errors_ = 0;
    double      bytes_ = 0;
    int         pending_inject_ = 0;
    std::vector<Inject> inj_;
    u64         detail_budget_ = 5000;  // всего строк с подробностями ошибок за прогон
    bool        budget_note_ = false, copy_note_ = false, stopped_on_error_ = false, ecc_fail_ = false;

    // статистика
    TestStat ts_[T_COUNT];
    u64      errors_ = 0, transient_ = 0, persistent_ = 0, unstable_ = 0;
    u64      bit_cnt_[64] = {};
    std::map<u64, AddrStat> addr_;
    u64      inj_total_ = 0, inj_hit_ = 0, inj_missed_ = 0;
    double   total_bytes_ = 0, bw_bytes_ = 0, bw_sec_ = 0;
    u64      bw_drops_ = 0;
    double   first_err_sec_ = -1;
    u64      first_err_loop_ = 0;
    const char* first_err_test_ = "";
    int      max_temp_ = -1, max_sm_ = -1, max_mem_ = -1, mem_clock_load_ = -1;
    double   max_power_ = -1;
    u64      limits_seen_ = 0;
};

// ---------------------------------- Подготовка --------------------------------------
void Tester::init_device() {
    LOG(L_MAIN, "gpumemtest " GMT_VERSION " - GPU video memory stress test with data error detection");
    LOG(L_INFO, "Command line: %s", o_.cmdline.c_str());
    int n = 0;
    CK(cudaGetDeviceCount(&n));
    if (o_.device >= n) throw std::runtime_error(strf("CUDA device %d not found (%d device(s) present)", o_.device, n));
    CK(cudaSetDevice(o_.device));
    CK(cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync));  // не крутить CPU в ожидании GPU

    cudaDeviceProp p;
    CK(cudaGetDeviceProperties(&p, o_.device));
    auto attr = [&](cudaDeviceAttr a) { int v = 0; cudaDeviceGetAttribute(&v, a, o_.device); return v; };
    di_.name = p.name;
    di_.total = p.totalGlobalMem;
    di_.cc_major = p.major;
    di_.cc_minor = p.minor;
    di_.sms = attr(cudaDevAttrMultiProcessorCount);
    di_.threads_per_sm = attr(cudaDevAttrMaxThreadsPerMultiProcessor);
    di_.bus_width = attr(cudaDevAttrGlobalMemoryBusWidth);
    di_.mem_clock_khz = attr(cudaDevAttrMemoryClockRate);
    di_.l2 = attr(cudaDevAttrL2CacheSize);
    di_.ecc = attr(cudaDevAttrEccEnabled);
    di_.tcc = attr(cudaDevAttrTccDriver);
    di_.watchdog = attr(cudaDevAttrKernelExecTimeout);
    di_.peak_bw = peak_bw(di_.mem_clock_khz / 1000);
    char pci[64] = {0};
    CK(cudaDeviceGetPCIBusId(pci, sizeof pci, o_.device));
    di_.pci = pci;
    CK(cudaDriverGetVersion(&di_.drv_ver));
    CK(cudaRuntimeGetVersion(&di_.rt_ver));
    CK(cudaFree(nullptr));  // создать контекст

    const bool nvml = nv_.init(pci);
    if (!di_.tcc) wddm_ok_ = wddm_.init(p.luid);
#ifdef _WIN32
    const char* model = di_.tcc ? "TCC" : "WDDM";
#else
    const char* model = "Linux";
#endif
    LOG(L_MAIN, "GPU %d: %s [%s], CC %d.%d, %d SMs, %s VRAM, %d-bit bus, nominal %d MHz = %.0f GB/s, L2 %s",
        o_.device, di_.name.c_str(), di_.pci.c_str(), di_.cc_major, di_.cc_minor, di_.sms,
        fmt_size((double)di_.total).c_str(), di_.bus_width, di_.mem_clock_khz / 1000, di_.peak_bw / 1e9,
        fmt_size(di_.l2).c_str());
    LOG(L_INFO, "Driver %s (CUDA %d.%d), runtime %d.%d, code built for sm %s, driver model %s, ECC %s, watchdog %s",
        nv_.driver.empty() ? "?" : nv_.driver.c_str(), di_.drv_ver / 1000, di_.drv_ver % 1000 / 10,
        di_.rt_ver / 1000, di_.rt_ver % 1000 / 10, GMT_ARCHS, model, di_.ecc ? "ON" : "off", di_.watchdog ? "on" : "off");
    if (nvml) LOG(L_INFO, "NVML telemetry: on (temperature, power, clocks, clock limit reasons)");
    else LOG(L_WARN, "NVML is not available - running without temperature/power/clock telemetry");
    u64 budget = 0, usage = 0;
    if (wddm_ok_ && wddm_.query(budget, usage))
        LOG(L_INFO, "WDDM video memory budget of this process: %s (in use %s) - the test stays below it to avoid "
                    "paging through system RAM", fmt_size((double)budget).c_str(), fmt_size((double)usage).c_str());
    if (di_.ecc)
        LOG(L_WARN, "ECC is ON: single-bit errors are corrected by hardware and stay invisible, uncorrectable "
                    "ones abort the test. To see raw errors disable ECC (NVIDIA Control Panel -> Change ECC State)");
    if (di_.watchdog)
        LOG(L_INFO, "Display watchdog (TDR) is active: every kernel launch processes one chunk (< 1 ms)");

    grid_ = di_.sms * std::max(1, di_.threads_per_sm / kBlock);
    CK(cudaMalloc(&d_hdr_, sizeof(ErrHdr)));
    CK(cudaMalloc(&d_rec_, kMaxRecords * sizeof(ErrRec)));
    CK(cudaMemset(d_hdr_, 0, sizeof(ErrHdr)));

    if (!o_.seed_set) {
        std::random_device rd;
        o_.seed = ((u64)rd() << 32) ^ rd() ^ ((u64)time(nullptr) << 16);
    }
    rng_.seed(o_.seed);
    inj_rng_.seed(mix64(o_.seed ^ 0x5EEDull));
    LOG(L_INFO, "Seed: 0x%016llx (reproduce patterns with --seed 0x%016llx)", o_.seed, o_.seed);
}

void Tester::reindex() {
    u64 off = 0;
    for (Chunk& c : ch_) { c.goff = off; c.gbase = off / 8; off += c.bytes; }
    region_ = off;
}

void Tester::allocate() {
    const u64 MiB = 1ull << 20;
    size_t freeb = 0, totalb = 0;
    CK(cudaMemGetInfo(&freeb, &totalb));
    // в WDDM рабочему столу и другим процессам нужен запас: бюджет "плавает" на 0.5-1 ГБ
    const u64 reserve = (o_.reserve_mb ? o_.reserve_mb : (wddm_ok_ ? 1024 : 256)) * MiB;
    u64 avail = freeb, budget = 0, usage = 0;
    const bool have_budget = wddm_ok_ && wddm_.query(budget, usage);
    if (have_budget) avail = std::min<u64>(avail, budget > usage ? budget - usage : 0);
    u64 target;
    if (o_.mem_mb) target = o_.mem_mb * MiB;
    else if (o_.mem_pct > 0) target = (u64)(avail * o_.mem_pct / 100.0);
    else target = avail > reserve ? avail - reserve : 0;
    const u64 hard_max = avail > 64 * MiB ? avail - 64 * MiB : 0;  // минимум 64 МБ всегда оставляем
    if (target > hard_max) {
        LOG(L_WARN, "Requested %s but only %s is available: testing %s", fmt_size((double)target).c_str(),
            fmt_size((double)avail).c_str(), fmt_size((double)hard_max).c_str());
        target = hard_max;
    }
    target &= ~(MiB - 1);
    LOG(L_INFO, "VRAM: total %s, free %s%s, reserve %s -> allocating %s", fmt_size((double)totalb).c_str(),
        fmt_size((double)freeb).c_str(), have_budget ? strf(", room in WDDM budget %s", fmt_size((double)avail).c_str()).c_str() : "",
        fmt_size((double)reserve).c_str(), fmt_size((double)target).c_str());

    u64 csize = o_.chunk_mb * MiB, got = 0;
    while (got < target && !g_stop) {
        const u64 want = std::min(csize, target - got);
        void* ptr = nullptr;
        const cudaError_t e = cudaMalloc(&ptr, want);
        if (e != cudaSuccess) {
            cudaGetLastError();  // ошибка выделения не "липкая" - сбрасываем
            if (e != cudaErrorMemoryAllocation) CK(e);
            if (want > 16 * MiB) { csize = std::max<u64>(16 * MiB, (want / 2) & ~(MiB - 1)); continue; }
            break;
        }
        ch_.push_back(Chunk{(u64*)ptr, want, want / 8, 0, 0});
        got += want;
    }
    reindex();
    if (ch_.empty()) throw std::runtime_error("could not allocate any video memory");
    if (got < target) LOG(L_WARN, "Allocated only %s of %s requested", fmt_size((double)got).c_str(), fmt_size((double)target).c_str());
    CK(cudaMemGetInfo(&freeb, &totalb));
    LOG(L_INFO, "Allocated %s in %zu chunk(s); VRAM still free: %s", fmt_size((double)got).c_str(), ch_.size(),
        fmt_size((double)freeb).c_str());
}

// Замер скорости записи каждого блока: в WDDM драйвер может молча разместить часть
// выделенной памяти в ОЗУ (sysmem fallback) - такие блоки тестировать бессмысленно.
void Tester::probe() {
    const Pat z = P_const(0);
    const auto t0 = clk::now();
    do {  // прогрев: память выходит из энергосберегающего режима, страницы становятся резидентными
        fill(z);
        CK(cudaDeviceSynchronize());
    } while (since(t0) < 0.5 && !g_stop);

    // Частота памяти под нагрузкой. При разгоне она выше номинала из CUDA, и пик
    // пропускной способности надо считать от неё.
    const int nominal = di_.mem_clock_khz / 1000, loaded = nv_.sample().mem;
    if (loaded > 0) {
        mem_clock_load_ = loaded;
        if (nominal > 0 && std::abs(loaded - nominal) > nominal / 50)
            LOG(L_INFO, "Memory clock under load: %d MHz vs nominal %d MHz (%+.1f%%)%s", loaded, nominal,
                100.0 * (loaded - nominal) / nominal, loaded > nominal ? " - memory is overclocked" : "");
    }
    const int peak_mhz = mem_clock_load_ > 0 ? mem_clock_load_ : nominal;

    cudaEvent_t ea, eb;
    CK(cudaEventCreate(&ea));
    CK(cudaEventCreate(&eb));
    std::vector<double> bw(ch_.size(), 0.0);
    for (size_t c = 0; c < ch_.size() && !g_stop; ++c)
        for (int r = 0; r < 3; ++r) {
            CK(cudaEventRecord(ea));
            fill_chunk(c, z);
            CK(cudaEventRecord(eb));
            CK(cudaEventSynchronize(eb));
            float ms = 0;
            CK(cudaEventElapsedTime(&ms, ea, eb));
            if (ms > 0) bw[c] = std::max(bw[c], ch_[c].bytes / (ms * 1e-3));
        }
    CK(cudaEventDestroy(ea));
    CK(cudaEventDestroy(eb));
    if (g_stop) return;

    std::vector<double> sorted = bw;
    std::sort(sorted.begin(), sorted.end());
    const double med = sorted[sorted.size() / 2];
    std::vector<Chunk> keep;
    u64 dropped = 0;
    for (size_t c = 0; c < ch_.size(); ++c) {
        if (bw[c] < 0.35 * med) {
            LOG(L_WARN, "Chunk %zu: write speed %.0f GB/s vs median %.0f GB/s - most likely placed in system RAM "
                        "(sysmem fallback), excluded", c, bw[c] / 1e9, med / 1e9);
            CK(cudaFree(ch_[c].ptr));
            dropped += ch_[c].bytes;
        } else {
            keep.push_back(ch_[c]);
        }
    }
    if (dropped) {
        ch_.swap(keep);
        reindex();
        LOG(L_WARN, "Excluded %s. Tip: NVIDIA Control Panel -> Manage 3D settings -> 'CUDA - Sysmem Fallback Policy' "
                    "= 'Prefer No Sysmem Fallback', or increase --reserve", fmt_size((double)dropped).c_str());
    }
    const double peak = peak_bw(peak_mhz);
    LOG(L_INFO, "Write bandwidth probe: median %.0f GB/s, min %.0f GB/s, max %.0f GB/s (%.0f%% of theoretical "
                "%.0f GB/s at MEM %d MHz)", med / 1e9, sorted.front() / 1e9, sorted.back() / 1e9,
        peak > 0 ? 100.0 * med / peak : 0.0, peak / 1e9, peak_mhz);
    if (peak > 0 && med < 0.3 * peak)
        LOG(L_WARN, "Bandwidth is far below the theoretical peak: memory is throttled, in a low power state "
                    "or partly located in system RAM");
    LOG(L_MAIN, "Testing %s = %.1f%% of VRAM in %zu chunk(s) of %s", fmt_size((double)region_).c_str(),
        100.0 * region_ / di_.total, ch_.size(), fmt_size((double)ch_[0].bytes).c_str());
    if (region_ < 4ull * di_.l2)
        LOG(L_WARN, "The tested region is smaller than 4x L2 cache (%s): many accesses will hit the cache, not DRAM",
            fmt_size(4.0 * di_.l2).c_str());
}

// ------------------------------------ Фазы ------------------------------------------
void Tester::fill_chunk(size_t c, const Pat& p) {
    const Chunk& k = ch_[c];
    const Pat q = for_chunk(p, k.gbase);
    DISPATCH(q.kind, k_fill, k.ptr, k.words / 2, k.gbase, q);
}

void Tester::fill(const Pat& p) {
    for (size_t c = 0; c < ch_.size(); ++c) fill_chunk(c, p);
    CK(cudaGetLastError());
    bytes_ += (double)region_;
}

// gexp - для каждого блока глобальный индекс, от которого считать ожидаемые данные
// (тест copy: в блоке-приёмнике лежат данные блока-источника)
u64 Tester::verify(const Pat& p, const std::string& what, const std::vector<u64>* gexp) {
    if (pending_inject_ > 0) {
        inject(pending_inject_);
        pending_inject_ = 0;
    }
    for (size_t c = 0; c < ch_.size(); ++c) {
        const Chunk& k = ch_[c];
        const u64 gb = gexp ? (*gexp)[c] : k.gbase;
        const Pat q = for_chunk(p, gb);
        const ErrCtx ec{d_hdr_, d_rec_, kMaxRecords, (unsigned)c};
        DISPATCH(q.kind, k_verify, k.ptr, k.words / 2, gb, q, ec);
    }
    CK(cudaGetLastError());
    bytes_ += (double)region_;
    return collect(what, true);
}

u64 Tester::verify_write(const Pat& p, u64 wmask, const std::string& what) {
    for (size_t c = 0; c < ch_.size(); ++c) {
        const Chunk& k = ch_[c];
        const Pat q = for_chunk(p, k.gbase);
        const ErrCtx ec{d_hdr_, d_rec_, kMaxRecords, (unsigned)c};
        DISPATCH(q.kind, k_verify_write, k.ptr, k.words / 2, k.gbase, q, wmask, ec);
    }
    CK(cudaGetLastError());
    bytes_ += 2.0 * region_;
    return collect(what, false);  // слова уже перезаписаны - перечитывать нечего
}

void Tester::mod_others(const Pat& p) {
    for (const Chunk& k : ch_) k_mod_others<<<grid_, kBlock>>>(k.ptr, k.words, for_chunk(p, k.gbase));
    CK(cudaGetLastError());
    bytes_ += (double)region_ * (p.m - 1) / p.m;
}

// Самопроверка: портим N случайных слов (по одному биту) прямо перед проверкой
void Tester::inject(int n) {
    CK(cudaDeviceSynchronize());  // запись шаблона должна завершиться
    for (int i = 0; i < n; ++i) {
        Inject j{};
        do {
            j.chunk = (unsigned)(inj_rng_() % ch_.size());
            j.idx = inj_rng_() % ch_[j.chunk].words;
        } while (std::any_of(inj_.begin(), inj_.end(),
                             [&](const Inject& q) { return q.chunk == j.chunk && q.idx == j.idx; }));
        CK(cudaMemcpy(&j.orig, ch_[j.chunk].ptr + j.idx, sizeof j.orig, cudaMemcpyDeviceToHost));
        j.value = j.orig ^ (1ull << (inj_rng_() % 64));
        CK(cudaMemcpy(ch_[j.chunk].ptr + j.idx, &j.value, sizeof j.value, cudaMemcpyHostToDevice));
        inj_.push_back(j);
    }
    inj_total_ += n;
    LOG(L_DEBUG, "self-test: %d bit flip(s) injected before the first check of '%s'", n, test_);
}

// Подробности об одном сбойном слове: где, что ожидалось, что прочитано, какие биты
std::string Tester::describe(const ErrRec& r) const {
    const Chunk& c = ch_[r.chunk];
    return strf("offset 0x%010llx (VA 0x%llx, chunk %u word %llu): expected 0x%016llx read 0x%016llx xor 0x%016llx %s",
                c.goff + r.idx * 8, (u64)(uintptr_t)(c.ptr + r.idx), r.chunk, r.idx, r.expected, r.actual,
                r.expected ^ r.actual, flip_str(r.expected, r.actual).c_str());
}

// Забрать результаты фазы с GPU, отделить внедрённые ошибки, записать журнал
u64 Tester::collect(const std::string& what, bool rereadable) {
    u64 cnt = 0;
    CK(cudaMemcpy(&cnt, &d_hdr_->count, sizeof cnt, cudaMemcpyDeviceToHost));  // заодно ждёт конца фазы
    std::vector<Inject> inj;
    inj.swap(inj_);
    if (cnt == 0 && inj.empty()) {
        LOG(L_DEBUG, "%s: %s - OK", test_, what.c_str());
        return 0;
    }

    ErrHdr h{};
    std::vector<ErrRec> rec;
    if (cnt) {
        CK(cudaMemcpy(&h, d_hdr_, sizeof h, cudaMemcpyDeviceToHost));
        rec.resize((size_t)std::min<u64>(cnt, kMaxRecords));
        CK(cudaMemcpy(rec.data(), d_rec_, rec.size() * sizeof(ErrRec), cudaMemcpyDeviceToHost));
        CK(cudaMemset(d_hdr_, 0, sizeof(ErrHdr)));
        std::sort(rec.begin(), rec.end(), [](const ErrRec& a, const ErrRec& b) {
            return a.chunk != b.chunk ? a.chunk < b.chunk : a.idx < b.idx;
        });
    }

    // внедрённые самопроверкой ошибки не считаем настоящими
    u64 hits = 0;
    std::vector<ErrRec> real;
    real.reserve(rec.size());
    for (const ErrRec& r : rec) {
        bool mine = false;
        for (Inject& j : inj)
            if (!j.hit && j.chunk == r.chunk && j.idx == r.idx && j.value == r.actual) { j.hit = mine = true; break; }
        if (!mine) { real.push_back(r); continue; }
        ++hits;
        LOG(L_INFO, "  injected and detected: %s", describe(r).c_str());
        for (u64 t = r.expected ^ r.actual; t; t &= t - 1)
            if (h.bit_cnt[ctz64(t)]) h.bit_cnt[ctz64(t)]--;
    }
    if (!inj.empty()) {
        inj_hit_ += hits;
        for (const Inject& j : inj) {
            if (!j.hit) {
                ++inj_missed_;
                LOG(L_ERROR, "SELF-TEST: injected bit flip in chunk %u word %llu was NOT detected (%s: %s)",
                    j.chunk, j.idx, test_, what.c_str());
            }
            // вернуть исходное значение, иначе следующие фазы примут внедрённый сбой за настоящий
            CK(cudaMemcpy(ch_[j.chunk].ptr + j.idx, &j.orig, sizeof j.orig, cudaMemcpyHostToDevice));
        }
        LOG(hits == inj.size() ? L_INFO : L_ERROR, "self-test: %llu of %zu injected bit flip(s) detected (%s: %s)",
            hits, inj.size(), test_, what.c_str());
    }
    const u64 n_real = cnt - hits;
    if (n_real == 0) return 0;

    errors_ += n_real;
    test_errors_ += n_real;
    if (first_err_sec_ < 0) { first_err_sec_ = since(t0_); first_err_loop_ = loop_; first_err_test_ = test_; }
    for (int b = 0; b < 64; ++b) bit_cnt_[b] += h.bit_cnt[b];
    u64 mask = 0;
    for (const ErrRec& r : real) mask |= r.expected ^ r.actual;
    if (cnt > rec.size()) mask |= h.bit_or;  // записаны не все ошибки - берём маску с GPU
    LOG(L_FAIL, "%s, loop %llu: %s -> %llu bad word(s)%s, flipped bits mask 0x%016llx", test_, loop_, what.c_str(),
        n_real, cnt > rec.size() ? strf(" (%zu recorded)", rec.size()).c_str() : "", mask);

    const size_t show = (size_t)std::min<u64>({(u64)real.size(), (u64)o_.max_report, detail_budget_});
    for (size_t i = 0; i < real.size(); ++i) {
        const ErrRec& r = real[i];
        const Chunk& c = ch_[r.chunk];
        const u64 off = c.goff + r.idx * 8;
        const u64 va = (u64)(uintptr_t)(c.ptr + r.idx);
        if (addr_.size() < kMaxAddrStats || addr_.count(off)) {
            AddrStat& a = addr_[off];
            if (!a.count) { a.first_loop = loop_; a.first_test = test_; a.va = va; }
            a.count++;
            a.bit_or |= r.expected ^ r.actual;
            a.last_loop = loop_;
        }
        if (i >= show) continue;
        std::string rr;
        if (rereadable) {  // к этому моменту L2 давно вытеснен: читаем из DRAM
            u64 v = 0;
            CK(cudaMemcpy(&v, c.ptr + r.idx, sizeof v, cudaMemcpyDeviceToHost));
            if (v == r.expected) { ++transient_; rr = " | re-read OK -> transient (read path)"; }
            else if (v == r.actual) { ++persistent_; rr = " | re-read same -> persistent (stored wrong)"; }
            else { ++unstable_; rr = strf(" | re-read 0x%016llx -> unstable", v); }
        }
        LOG(L_ERROR, "  %s%s", describe(r).c_str(), rr.c_str());
    }
    detail_budget_ -= show;
    if (n_real > show) LOG(L_ERROR, "  ... %llu more bad word(s) in this phase not listed", n_real - show);
    if (detail_budget_ == 0 && !budget_note_) {
        budget_note_ = true;
        LOG(L_WARN, "Detail limit reached: further errors are counted but not listed one by one");
    }
    if (o_.stop_on_error) { stopped_on_error_ = true; g_stop = true; }
    return n_real;
}

// ------------------------------------ Тесты -----------------------------------------
// Каждый тест проверяет остановку между фазами и возвращает true, если дошёл до конца.
bool Tester::t_addr() {
    for (int inv = 0; inv < 2; ++inv) {
        if (should_stop()) return false;
        const Pat p = P_addr(inv ? ~0ull : 0ull);
        fill(p);
        verify(p, inv ? "W/R ~address" : "W/R address");
    }
    return true;
}

bool Tester::t_random() {
    for (int i = 0; i < o_.rand_passes; ++i) {
        if (should_stop()) return false;
        const u64 seed = rng_();
        const Pat p = P_rand(seed, 0);
        fill(p);
        verify(p, strf("W/R random, seed %016llx", seed));
        if (should_stop()) return false;
        verify_write(p, ~0ull, strf("R(rnd)W(~rnd), seed %016llx", seed));
        verify(P_rand(seed, ~0ull), strf("R(~rnd), seed %016llx", seed));
    }
    return true;
}

bool Tester::walk(u64 base, const char* name) {
    for (unsigned s = 0; s < 64; ++s) {
        if (should_stop()) return false;
        const Pat p = P_walk(base, s);
        fill(p);
        verify(p, strf("%s, shift %u/64", name, s + 1));
    }
    return true;
}

bool Tester::t_movinv() {
    const u64 pats[] = {0x0000000000000000ull, 0xAAAAAAAAAAAAAAAAull, 0xCCCCCCCCCCCCCCCCull,
                        0xF0F0F0F0F0F0F0F0ull, 0xFF00FF00FF00FF00ull, 0xFFFF0000FFFF0000ull,
                        0xFFFFFFFF00000000ull, rng_()};
    for (const u64 v : pats) {  // ~P для каждого шаблона проверяется внутри элемента
        if (should_stop()) return false;
        const Pat p = P_const(v);
        fill(p);
        verify_write(p, ~0ull, strf("P=%016llx: R(P)W(~P)", v));
        verify_write(P_const(~v), ~0ull, strf("P=%016llx: R(~P)W(P)", v));
        verify(p, strf("P=%016llx: R(P)", v));
    }
    return true;
}

bool Tester::t_mod20() {
    const unsigned M = 20;
    const u64 base = rng_();
    for (unsigned k = 0; k < M; ++k)
        for (int inv = 0; inv < 2; ++inv) {
            if (should_stop()) return false;
            const u64 v = inv ? ~base : base;
            const Pat p = P_mod(v, M, k);
            fill(p);
            mod_others(p);
            mod_others(p);
            verify(p, strf("offset %u/%u, P=%016llx", k + 1, M, v));
        }
    return true;
}

bool Tester::t_copy() {
    const size_t n = ch_.size(), half = n / 2;
    if (half == 0) {
        if (!copy_note_) { copy_note_ = true; LOG(L_WARN, "copy: needs at least 2 chunks - skipped (use a smaller --chunk)"); }
        return true;
    }
    for (int dir = 0; dir < 2; ++dir) {
        if (should_stop()) return false;
        const u64 seed = rng_();
        const Pat p = P_rand(seed, 0);
        fill(p);
        std::vector<u64> gexp(n);
        for (size_t c = 0; c < n; ++c) gexp[c] = ch_[c].gbase;
        double copied = 0;
        for (size_t j = 0; j < half; ++j) {
            const size_t s = dir ? j + half : j, d = dir ? j : j + half;
            if (ch_[s].bytes != ch_[d].bytes) continue;
            CK(cudaMemcpyAsync(ch_[d].ptr, ch_[s].ptr, ch_[s].bytes, cudaMemcpyDeviceToDevice, 0));
            gexp[d] = ch_[s].gbase;
            copied += (double)ch_[s].bytes;
        }
        bytes_ += 2 * copied;
        verify(p, strf("%s, seed %016llx", dir ? "copy 2nd half -> 1st half" : "copy 1st half -> 2nd half", seed), &gexp);
    }
    return true;
}

bool Tester::t_stress() {
    const u64 seed = rng_();
    Pat p = P_rand(seed, 0);
    fill(p);
    bool done = true;
    for (int i = 0; i < o_.stress_iters; ++i) {
        if (should_stop()) { done = false; break; }
        verify_write(p, ~0ull, strf("sweep %d/%d, seed %016llx", i + 1, o_.stress_iters, seed));
        p.b = ~p.b;  // теперь в памяти инверсия
    }
    verify(p, strf("final check, seed %016llx", seed));
    return done;
}

bool Tester::t_fade() {
    for (const u64 v : {0ull, ~0ull}) {
        if (should_stop()) return false;
        const Pat p = P_const(v);
        fill(p);
        CK(cudaDeviceSynchronize());
        LOG(L_INFO, "fade: pattern %016llx written, GPU idle for %d s", v, o_.fade_sec);
        const auto t0 = clk::now();
        while (!should_stop() && since(t0) < o_.fade_sec) std::this_thread::sleep_for(std::chrono::milliseconds(100));
        verify(p, strf("P=%016llx after %.0f s idle", v, since(t0)));
    }
    return true;
}

// ---------------------------------- Прогон ------------------------------------------
// Windows уменьшила бюджет процесса (память понадобилась другим приложениям): отдаём
// блоки с конца области, пока не вернёмся под бюджет с запасом. Смещения оставшихся
// блоков не меняются, статистика по адресам остаётся верной.
void Tester::check_budget() {
    const u64 MiB = 1ull << 20;
    u64 budget = 0, usage = 0, freed = 0;
    if (!wddm_ok_ || !wddm_.query(budget, usage) || usage + 64 * MiB <= budget) return;
    while (ch_.size() > 1 && usage + 256 * MiB > budget + freed) {
        freed += ch_.back().bytes;
        CK(cudaFree(ch_.back().ptr));
        ch_.pop_back();
    }
    if (!freed) return;
    reindex();
    shrunk_ += freed;
    LOG(L_WARN, "WDDM budget of this process dropped to %s (in use %s): released %s to avoid paging through "
                "system RAM, now testing %s", fmt_size((double)budget).c_str(), fmt_size((double)usage).c_str(),
        fmt_size((double)freed).c_str(), fmt_size((double)region_).c_str());
}

void Tester::note_telemetry(const Nvml::Sample& s) {
    if (!s.ok) return;
    max_temp_ = std::max(max_temp_, s.temp);
    max_power_ = std::max(max_power_, s.power);
    max_sm_ = std::max(max_sm_, s.sm);
    max_mem_ = std::max(max_mem_, s.mem);
    limits_seen_ |= s.reasons & ~1ull;  // без "GPU idle"
}

void Tester::set_title() {
#ifdef _WIN32
    SetConsoleTitleA(strf("gpumemtest - loop %llu - %llu error(s) - %s", loop_, errors_, fmt_dur(since(t0_)).c_str()).c_str());
#endif
}

bool Tester::run_test(int id) {
    const TestDef& t = kTests[id];
    check_budget();
    test_ = t.name;
    test_errors_ = 0;
    bytes_ = 0;
    pending_inject_ = o_.inject;
    const auto t0 = clk::now();
    bool done = false;
    switch (id) {
    case T_ADDR:   done = t_addr(); break;
    case T_RANDOM: done = t_random(); break;
    case T_WALK1:  done = walk(1ull, "walking 1"); break;
    case T_WALK0:  done = walk(~1ull, "walking 0"); break;
    case T_MOVINV: done = t_movinv(); break;
    case T_MOD20:  done = t_mod20(); break;
    case T_COPY:   done = t_copy(); break;
    case T_STRESS: done = t_stress(); break;
    case T_FADE:   done = t_fade(); break;
    }
    CK(cudaDeviceSynchronize());
    const double sec = since(t0);

    TestStat& s = ts_[id];
    s.errors += test_errors_;
    s.sec += sec;
    s.bytes += bytes_;
    total_bytes_ += bytes_;
    if (t.has_bw) { bw_bytes_ += bytes_; bw_sec_ += sec; }
    if (done) s.runs++;

    const Nvml::Sample tm = nv_.sample();
    note_telemetry(tm);
    const double bw = sec > 0 ? bytes_ / sec / 1e9 : 0;
    const std::string bws = t.has_bw ? strf("%.0f GB/s", bw) : std::string("-");
    const std::string tel = tele_str(tm);
    if (test_errors_)
        LOG(L_FAIL, "loop %llu | %-6s | %7.2f s | %9s | %llu ERROR(S) | %s", loop_, t.name, sec, bws.c_str(), test_errors_, tel.c_str());
    else
        LOG(done ? L_PASS : L_INFO, "loop %llu | %-6s | %7.2f s | %9s | %s | %s", loop_, t.name, sec, bws.c_str(),
            done ? "0 errors" : "interrupted", tel.c_str());

    // Падение скорости при тех же частотах - троттлинг, вытеснение VRAM в ОЗУ или
    // повторы передачи GDDR6X (EDC) при нестабильном разгоне памяти.
    if (done && t.has_bw && bw > 0) {
        if (s.base_bw == 0) {
            s.base_bw = bw;
        } else if (!s.slow && bw < 0.75 * s.base_bw) {
            s.slow = true;
            ++bw_drops_;
            LOG(L_WARN, "%s: bandwidth dropped to %.0f GB/s (first loop %.0f GB/s): throttling, VRAM demoted to "
                        "system RAM, or GDDR6X EDC retries (unstable memory overclock)", t.name, bw, s.base_bw);
        } else if (s.slow && bw > 0.9 * s.base_bw) {
            s.slow = false;
            LOG(L_INFO, "%s: bandwidth recovered to %.0f GB/s", t.name, bw);
        }
    }
    set_title();
    return done;
}

std::string Tester::stop_reason() const {
    if (stopped_on_error_) return "first error (--stop-on-error)";
    if (g_closing) return "console window closed";
    if (g_stop) return "user (Ctrl+C)";
    if (has_deadline_ && clk::now() >= deadline_) return "time limit";
    if (o_.loops && loops_done_ >= o_.loops) return "loop limit";
    return "end";
}

int Tester::run() {
    std::string names;
    for (int id : o_.tests) names += (names.empty() ? "" : ",") + std::string(kTests[id].name);
    LOG(L_MAIN, "Tests per loop: %s", names.c_str());
    for (int id : o_.tests) LOG(L_INFO, "  %-6s - %s", kTests[id].name, kTests[id].title);
    if (o_.inject) LOG(L_WARN, "SELF-TEST mode: %d bit flip(s) are injected before the first check of every test", o_.inject);
    std::string lim;
    if (o_.time_sec > 0) lim = "for " + fmt_dur(o_.time_sec);
    if (o_.loops) lim += (lim.empty() ? "for " : " or ") + strf("%llu loop(s)", o_.loops);
    LOG(L_MAIN, "Running %s. Press Ctrl+C to stop - the summary is written on exit.", lim.empty() ? "until stopped" : lim.c_str());

    t0_ = clk::now();
    started_ = true;
    if (o_.time_sec > 0) {
        has_deadline_ = true;
        deadline_ = t0_ + std::chrono::duration_cast<clk::duration>(std::chrono::duration<double>(o_.time_sec));
    }
    for (loop_ = 1; !should_stop(); ++loop_) {
        const auto lt = clk::now();
        const u64 e0 = errors_;
        bool complete = true;
        for (int id : o_.tests)
            if (should_stop() || !run_test(id)) { complete = false; break; }
        if (!complete) {
            LOG(L_MAIN, "Loop %llu interrupted after %s", loop_, fmt_dur(since(lt)).c_str());
            break;
        }
        loops_done_ = loop_;
        const u64 le = errors_ - e0;
        const std::string left = has_deadline_
            ? strf(", left %s", fmt_dur(std::max(0.0, std::chrono::duration<double>(deadline_ - clk::now()).count())).c_str())
            : std::string();
        LOG(le ? L_FAIL : L_MAIN, "=== loop %llu done in %s: %s | total errors: %llu | elapsed %s%s", loop_,
            fmt_dur(since(lt)).c_str(), le ? strf("%llu ERROR(S)", le).c_str() : "no errors", errors_,
            fmt_dur(since(t0_)).c_str(), left.c_str());
        if (o_.loops && loops_done_ >= o_.loops) break;
    }
    const std::string reason = stop_reason();
    CK(cudaDeviceSynchronize());
    summary(reason, false);
    return (errors_ || inj_missed_) ? 1 : 0;
}

void Tester::summary(const std::string& reason, bool aborted) {
    const double el = since(t0_);
    LOG(L_MAIN, "=============================== SUMMARY ===============================");
    LOG(L_MAIN, "GPU         : %s [%s]", di_.name.c_str(), di_.pci.c_str());
    LOG(L_MAIN, "Tested      : %s in %zu chunk(s) = %.1f%% of %s VRAM%s", fmt_size((double)region_).c_str(), ch_.size(),
        di_.total ? 100.0 * region_ / di_.total : 0.0, fmt_size((double)di_.total).c_str(),
        shrunk_ ? strf(" (started with %s, reduced to fit the WDDM budget)", fmt_size((double)(region_ + shrunk_)).c_str()).c_str() : "");
    LOG(L_MAIN, "Run time    : %s, full loops: %llu, stopped by: %s", fmt_dur(el).c_str(), loops_done_, reason.c_str());
    const double avg = bw_sec_ > 0 ? bw_bytes_ / bw_sec_ : 0.0;
    const int peak_mhz = max_mem_ > 0 ? max_mem_ : di_.mem_clock_khz / 1000;
    const double peak = peak_bw(peak_mhz);
    LOG(L_MAIN, "Traffic     : %s read+written, average %.0f GB/s = %.0f%% of theoretical %.0f GB/s at MEM %d MHz",
        fmt_traffic(total_bytes_).c_str(), avg / 1e9, peak > 0 ? 100.0 * avg / peak : 0.0, peak / 1e9, peak_mhz);
    if (max_temp_ >= 0 || max_power_ >= 0) {
        const std::string lim = limits_str(limits_seen_);
        LOG(L_MAIN, "Telemetry   : max GPU temp %d C, max power %.0f W, max clocks SM %d / MEM %d MHz, clock limits: %s",
            max_temp_, max_power_, max_sm_, max_mem_, lim.empty() ? "none" : lim.c_str());
    }
    if (bw_drops_) LOG(L_WARN, "Bandwidth   : dropped below 75%% of the first loop %llu time(s), see WARN lines", bw_drops_);
    LOG(L_MAIN, "  %-7s %6s %12s %12s %10s", "test", "runs", "errors", "time", "avg BW");
    for (int id : o_.tests) {
        const TestStat& s = ts_[id];
        const std::string bw = kTests[id].has_bw && s.sec > 0 ? strf("%.0f GB/s", s.bytes / s.sec / 1e9) : std::string("-");
        LOG(s.errors ? L_FAIL : L_MAIN, "  %-7s %6llu %12llu %12s %10s", kTests[id].name, s.runs, s.errors,
            fmt_dur(s.sec).c_str(), bw.c_str());
    }
    if (inj_total_)
        LOG(inj_missed_ ? L_ERROR : L_GOOD, "Self-test   : %llu bit flip(s) injected, %llu detected, %llu missed -> %s",
            inj_total_, inj_hit_, inj_missed_, inj_missed_ ? "ERROR DETECTION FAILED" : "error detection works");

    if (errors_) {
        LOG(L_FAIL, "Errors      : %llu bad word(s); first at %s (loop %llu, test %s)", errors_,
            fmt_dur(first_err_sec_).c_str(), first_err_loop_, first_err_test_);
        if (transient_ + persistent_ + unstable_)
            LOG(L_FAIL, "Re-read     : %llu transient (memory held the right value), %llu persistent (wrong value "
                        "stored), %llu unstable", transient_, persistent_, unstable_);
        std::string hist;
        int nb = 0;
        for (int b = 0; b < 64; ++b) {
            if (!bit_cnt_[b]) continue;
            hist += strf(" b%d:%llu", b, bit_cnt_[b]);
            if (++nb % 12 == 0) { LOG(L_FAIL, "Failing bits:%s", hist.c_str()); hist.clear(); }
        }
        if (!hist.empty()) LOG(L_FAIL, "Failing bits:%s", hist.c_str());
        std::vector<std::pair<u64, AddrStat>> v(addr_.begin(), addr_.end());
        std::sort(v.begin(), v.end(), [](const std::pair<u64, AddrStat>& a, const std::pair<u64, AddrStat>& b) {
            return a.second.count != b.second.count ? a.second.count > b.second.count : a.first < b.first;
        });
        LOG(L_FAIL, "Failing addresses: %zu%s unique, most frequent:", addr_.size(), addr_.size() >= kMaxAddrStats ? "+" : "");
        for (size_t i = 0; i < v.size() && i < 20; ++i)
            LOG(L_FAIL, "  offset 0x%010llx (VA 0x%llx): %llu time(s), bits 0x%016llx, loops %llu..%llu, first in %s",
                v[i].first, v[i].second.va, v[i].second.count, v[i].second.bit_or, v[i].second.first_loop,
                v[i].second.last_loop, v[i].second.first_test);
        LOG(L_FAIL, "RESULT: FAIL - %llu memory error(s) detected", errors_);
    } else if (ecc_fail_) {
        LOG(L_FAIL, "RESULT: FAIL - the GPU reported an uncorrectable ECC error");
    } else if (aborted) {
        LOG(L_ERROR, "RESULT: ABORTED (%s); no data errors were detected before that", reason.c_str());
    } else if (inj_missed_) {
        LOG(L_ERROR, "RESULT: FAIL - the self-test missed injected errors");
    } else if (loops_done_ == 0) {
        LOG(L_WARN, "RESULT: no errors detected, but no full loop was completed");
    } else {
        LOG(L_GOOD, "RESULT: PASS - no memory errors detected (%llu full loop(s), %s)", loops_done_, fmt_dur(el).c_str());
    }
}

int Tester::fatal(const CudaFatal& f) {
    LOG(L_ERROR, "CUDA error %s: %s [%s]", cudaGetErrorName(f.err), cudaGetErrorString(f.err), f.where.c_str());
    switch (f.err) {
    case cudaErrorECCUncorrectable:
        ecc_fail_ = true;
        LOG(L_ERROR, "The GPU reported an UNCORRECTABLE ECC error - this is a video memory failure");
        break;
    case cudaErrorNoKernelImageForDevice:
    case cudaErrorUnsupportedPtxVersion:
        LOG(L_ERROR, "This build has no code for this GPU: rebuild with -arch=sm_%d%d or update the driver",
            di_.cc_major, di_.cc_minor);
        break;
    case cudaErrorInsufficientDriver:
    case cudaErrorNoDevice:
        LOG(L_ERROR, "No usable NVIDIA GPU/driver: install or update the NVIDIA driver");
        break;
    case cudaErrorLaunchTimeout:
        LOG(L_ERROR, "A kernel was stopped by the display watchdog (TDR)");
        break;
    default:
        if (started_)
            LOG(L_ERROR, "The GPU context was lost (driver reset / Xid). Under this load it is itself a sign of "
                         "instability: memory or core overclock, overheating, power delivery");
        break;
    }
    if (started_) {
        LOG(L_ERROR, "Check Windows Event Viewer -> System -> source 'nvlddmkm' (Linux: dmesg | grep -i xid)");
        summary(strf("CUDA error %s", cudaGetErrorName(f.err)), true);
    }
    return (ecc_fail_ || errors_) ? 1 : 2;
}

void Tester::release() {
    for (const Chunk& c : ch_) cudaFree(c.ptr);
    ch_.clear();
    if (d_hdr_) cudaFree(d_hdr_);
    if (d_rec_) cudaFree(d_rec_);
    d_hdr_ = nullptr;
    d_rec_ = nullptr;
}

// =====================================================================================
int main(int argc, char** argv) {
    Options o;
    try {
        const int rc = parse_args(argc, argv, o);
        if (rc >= 0) return rc;
    } catch (const std::invalid_argument& e) {
        fprintf(stderr, "Error: %s\nUse --help for the list of options.\n", e.what());
        return 3;
    }
    if (o.list) return list_devices();

#ifdef _WIN32
    SetConsoleCtrlHandler(on_console_ctrl, TRUE);
    SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED);  // не уходить в сон во время теста
#else
    signal(SIGINT, on_signal);
    signal(SIGTERM, on_signal);
#endif

    g_logger.color = o.color && console_supports_color();
    g_logger.quiet = o.quiet;
    g_logger.verbose = o.verbose;
    if (o.log_path.empty()) o.log_path = "gpumemtest_" + file_stamp() + ".log";
    if (!g_logger.open(o.log_path)) {
        fprintf(stderr, "Error: cannot open log file '%s'\n", o.log_path.c_str());
        return 3;
    }
    LOG(L_INFO, "Log file: %s", o.log_path.c_str());

    Nvml nv;
    Tester t(o, nv);
    int rc = 2;
    bool context_ok = true;
    try {
        t.init_device();
        t.allocate();
        t.probe();
        rc = t.run();
    } catch (const CudaFatal& f) {
        rc = t.fatal(f);
        context_ok = false;  // после "липкой" ошибки контекст мёртв - память освободит драйвер
    } catch (const std::exception& e) {
        LOG(L_ERROR, "Fatal: %s", e.what());
        rc = 2;
    }
    if (context_ok) t.release();
    nv.shutdown();
    LOG(L_INFO, "Exit code %d. Log saved to %s", rc, o.log_path.c_str());
    g_logger.close();
    g_done = true;

#ifdef _WIN32
    SetThreadExecutionState(ES_CONTINUOUS);
    if (o.pause && !g_closing && own_console()) {
        printf("\nPress Enter to close this window...");
        fflush(stdout);
        getchar();
    }
#endif
    return rc;
}
