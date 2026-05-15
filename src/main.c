/*
 * AcoustEEEcare — SD Card Isolated Test
 * main.c
 *
 * Mirrors the full 6-step init_sd_card() sequence from AcoustEEEcare v6.6,
 * plus a write → read → verify probe roundtrip, all without BLE or audio.
 *
 * LED status (matches AcoustEEEcare v6.6 scheme):
 *   White         Boot / initialising
 *   Cyan          SD init in progress (steps 1–6)
 *   Yellow flash  Non-fatal warning (retry / statvfs fail / card full)
 *   Green         All tests passed
 *   Red blink     Fatal error — loops forever, read serial log
 *
 * XIAO BLE Sense onboard RGB LED GPIO (active-low, all on gpio0):
 *   Red   → P0.26
 *   Green → P0.30
 *   Blue  → P0.06
 *
 * Build:
 *   west build -p always -b xiao_ble/nrf52840/sense -- \
 *       -DDTC_OVERLAY_FILE=xiao_ble_nrf52840_sense.overlay
 *
 * Flash: double-tap RESET to enter UF2 bootloader, copy zephyr.uf2
 *
 * Serial logs: open the XIAO's CMSIS-DAP COM port in any serial monitor.
 *   Logs appear immediately at boot — no special baud rate needed.
 */

#include <zephyr/kernel.h>
#include <zephyr/logging/log.h>
#include <zephyr/drivers/gpio.h>
#include <zephyr/storage/disk_access.h>
#include <zephyr/fs/fs.h>
#include <ff.h>
#include <string.h>

LOG_MODULE_REGISTER(main, LOG_LEVEL_INF);

/* ------------------------------------------------------------------ */
/*  LED — XIAO BLE Sense onboard RGB (active-low)                     */
/* ------------------------------------------------------------------ */

#define LED_R_PIN  26   /* P0.26 — Red   */
#define LED_G_PIN  30   /* P0.30 — Green */
#define LED_B_PIN   6   /* P0.06 — Blue  */

static const struct device *led_gpio;

/* val=1 → channel ON, val=0 → OFF (active-low inversion here) */
static inline void led_r(int val) { gpio_pin_set_raw(led_gpio, LED_R_PIN, !val); }
static inline void led_g(int val) { gpio_pin_set_raw(led_gpio, LED_G_PIN, !val); }
static inline void led_b(int val) { gpio_pin_set_raw(led_gpio, LED_B_PIN, !val); }

static void led_off(void)    { led_r(0); led_g(0); led_b(0); }
static void led_white(void)  { led_r(1); led_g(1); led_b(1); }   /* Boot           */
static void led_cyan(void)   { led_r(0); led_g(1); led_b(1); }   /* SD working     */
static void led_green(void)  { led_r(0); led_g(1); led_b(0); }   /* Pass           */
static void led_red(void)    { led_r(1); led_g(0); led_b(0); }   /* Fatal error    */
static void led_yellow(void) { led_r(1); led_g(1); led_b(0); }   /* Step warning   */

/* Three yellow flashes then back to cyan — non-fatal step warning */
static void led_warn_flash(void)
{
    for (int i = 0; i < 3; i++) {
        led_yellow();
        k_msleep(120);
        led_off();
        k_msleep(80);
    }
    led_cyan();
}

/* Rapid red blink forever — call on unrecoverable failure */
static void led_error_blink(void)
{
    LOG_ERR("Entering error blink loop (Red) — read serial log for details");
    while (1) {
        led_red();
        k_msleep(200);
        led_off();
        k_msleep(200);
    }
}

static int led_init(void)
{
    led_gpio = DEVICE_DT_GET(DT_NODELABEL(gpio0));
    if (!device_is_ready(led_gpio)) {
        return -ENODEV;
    }
    gpio_pin_configure(led_gpio, LED_R_PIN, GPIO_OUTPUT_INACTIVE);
    gpio_pin_configure(led_gpio, LED_G_PIN, GPIO_OUTPUT_INACTIVE);
    gpio_pin_configure(led_gpio, LED_B_PIN, GPIO_OUTPUT_INACTIVE);
    return 0;
}

/* ------------------------------------------------------------------ */
/*  Constants                                                          */
/* ------------------------------------------------------------------ */

#define DISK_NAME          "SD"
#define MOUNT_POINT        "/SD:"
#define PROBE_FILE         MOUNT_POINT "/probe.tmp"
#define TEST_FILE          MOUNT_POINT "/sd_test.pcm"

#define INIT_RETRIES       5
#define INIT_RETRY_MS      1500
#define TEST_PAYLOAD_BYTES 512

/* ------------------------------------------------------------------ */
/*  Static storage                                                     */
/* ------------------------------------------------------------------ */

static FATFS fat_fs;

static struct fs_mount_t mp = {
    .type      = FS_FATFS,
    .fs_data   = &fat_fs,   /* v6.6 fix: struct pointer, not FS_FATFS constant */
    .mnt_point = MOUNT_POINT,
};

static uint8_t wr_buf[TEST_PAYLOAD_BYTES];
static uint8_t rd_buf[TEST_PAYLOAD_BYTES];

/* ------------------------------------------------------------------ */
/*  Helper: 8-bit XOR checksum                                        */
/* ------------------------------------------------------------------ */
static uint8_t xor_checksum(const uint8_t *buf, size_t len)
{
    uint8_t cs = 0;
    for (size_t i = 0; i < len; i++) {
        cs ^= buf[i];
    }
    return cs;
}

/* ------------------------------------------------------------------ */
/*  init_sd_card — 6-step sequence, mirrors AcoustEEEcare v6.6        */
/* ------------------------------------------------------------------ */
static int init_sd_card(void)
{
    int ret;

    /* Step 1: disk_access_init with retries ----------------------- */
    LOG_INF("[1/6] Initialising disk '%s' ...", DISK_NAME);
    for (int attempt = 0; attempt < INIT_RETRIES; attempt++) {
        ret = disk_access_init(DISK_NAME);
        if (ret == 0) {
            break;
        }
        LOG_WRN("  attempt %d/%d failed (%d), retrying ...",
                attempt + 1, INIT_RETRIES, ret);
        led_warn_flash();

        LOG_INF("Waiting 250ms for SD power rail to stabilise...");
        k_msleep(250);
        k_msleep(INIT_RETRY_MS);
    }
    if (ret != 0) {
        LOG_ERR("[1/6] FAIL: disk_access_init returned %d after %d attempts",
                ret, INIT_RETRIES);
        return ret;
    }
    LOG_INF("[1/6] disk_access_init OK");

    /* Step 2: sector count ---------------------------------------- */
    uint32_t sector_count = 0;
    ret = disk_access_ioctl(DISK_NAME, DISK_IOCTL_GET_SECTOR_COUNT, &sector_count);
    if (ret != 0 || sector_count == 0) {
        LOG_ERR("[2/6] FAIL: sector count ioctl ret=%d count=%u", ret, sector_count);
        return (ret != 0) ? ret : -EIO;
    }
    LOG_INF("[2/6] sector_count=%u  (~%u MiB)",
            sector_count,
            (uint32_t)((uint64_t)sector_count * 512 / (1024 * 1024)));

    /* Step 3: sector size ----------------------------------------- */
    uint32_t sector_size = 0;
    ret = disk_access_ioctl(DISK_NAME, DISK_IOCTL_GET_SECTOR_SIZE, &sector_size);
    if (ret != 0 || sector_size != 512) {
        LOG_ERR("[3/6] FAIL: sector size ret=%d size=%u (expected 512)",
                ret, sector_size);
        return (ret != 0) ? ret : -EIO;
    }
    LOG_INF("[3/6] sector_size=%u", sector_size);

    /* Step 4: mount FAT filesystem -------------------------------- */
    ret = fs_mount(&mp);
    if (ret != 0) {
        LOG_ERR("[4/6] FAIL: fs_mount returned %d", ret);
        int sync_ret = disk_access_ioctl(DISK_NAME, DISK_IOCTL_CTRL_SYNC, NULL);
        LOG_INF("  CTRL_SYNC probe: %s",
                (sync_ret == 0)
                    ? "SPI OK — check card format (exFAT needs CONFIG_FS_FATFS_EXFAT=y)"
                    : "SPI also failing — check wiring/power");
        return ret;
    }
    LOG_INF("[4/6] FAT mount OK  (%s)", MOUNT_POINT);

    /* Step 5: free space ------------------------------------------ */
    struct fs_statvfs sbuf;
    ret = fs_statvfs(MOUNT_POINT, &sbuf);
    if (ret != 0) {
        LOG_WRN("[5/6] fs_statvfs failed (%d) — continuing anyway", ret);
        led_warn_flash();
    } else {
        uint64_t free_bytes = (uint64_t)sbuf.f_bfree * sbuf.f_frsize;
        LOG_INF("[5/6] free=%llu MiB",
                (unsigned long long)(free_bytes / (1024 * 1024)));
        if (sbuf.f_bfree == 0) {
            LOG_WRN("  WARNING: SD card is full — writes will fail");
            led_warn_flash();
        }
    }

    /* Step 6: probe write → read → verify → unlink --------------- */
    LOG_INF("[6/6] Probe write/read roundtrip on %s ...", PROBE_FILE);

    static const char probe_payload[] = "AcoustEEEcare-SD-probe-v6.6\n";
    char probe_rd[sizeof(probe_payload)];

    struct fs_file_t probe;
    fs_file_t_init(&probe);

    ret = fs_open(&probe, PROBE_FILE, FS_O_CREATE | FS_O_RDWR);
    if (ret != 0) {
        LOG_ERR("[6/6] FAIL: fs_open probe returned %d", ret);
        return ret;
    }
    ret = fs_write(&probe, probe_payload, sizeof(probe_payload) - 1);
    if (ret < 0) {
        LOG_ERR("[6/6] FAIL: fs_write probe returned %d", ret);
        fs_close(&probe);
        return ret;
    }
    fs_seek(&probe, 0, FS_SEEK_SET);
    ret = fs_read(&probe, probe_rd, sizeof(probe_payload) - 1);
    if (ret < 0) {
        LOG_ERR("[6/6] FAIL: fs_read probe returned %d", ret);
        fs_close(&probe);
        return ret;
    }
    fs_close(&probe);
    fs_unlink(PROBE_FILE);

    if (memcmp(probe_payload, probe_rd, sizeof(probe_payload) - 1) != 0) {
        LOG_ERR("[6/6] FAIL: probe read-back mismatch — data corruption");
        return -EIO;
    }
    LOG_INF("[6/6] probe write+read+verify OK  → SD is writable");

    return 0;
}

/* ------------------------------------------------------------------ */
/*  run_payload_test — 512-byte ramp write, verify, XOR checksum      */
/* ------------------------------------------------------------------ */
static int run_payload_test(void)
{
    for (int i = 0; i < TEST_PAYLOAD_BYTES; i++) {
        wr_buf[i] = (uint8_t)(i & 0xFF);
    }
    uint8_t expected_cs = xor_checksum(wr_buf, TEST_PAYLOAD_BYTES);

    LOG_INF("Writing %d-byte ramp payload to %s ...", TEST_PAYLOAD_BYTES, TEST_FILE);

    struct fs_file_t f;
    fs_file_t_init(&f);

    int ret = fs_open(&f, TEST_FILE, FS_O_CREATE | FS_O_RDWR | FS_O_TRUNC);
    if (ret != 0) {
        LOG_ERR("fs_open %s failed (%d)", TEST_FILE, ret);
        return ret;
    }

    ret = fs_write(&f, wr_buf, TEST_PAYLOAD_BYTES);
    if (ret < 0) {
        LOG_ERR("fs_write failed (%d)", ret);
        fs_close(&f);
        return ret;
    }
    LOG_INF("Wrote %d bytes to %s", ret, TEST_FILE);

    /* Append 4-byte XOR checksum (matches AcoustEEEcare file format) */
    uint8_t cs_block[4] = { expected_cs, 0x00, 0x00, 0x00 };
    ret = fs_write(&f, cs_block, sizeof(cs_block));
    if (ret < 0) {
        LOG_ERR("fs_write checksum block failed (%d)", ret);
        fs_close(&f);
        return ret;
    }
    LOG_INF("XOR checksum appended (0x%02X)", expected_cs);

    /* Read back and verify */
    fs_seek(&f, 0, FS_SEEK_SET);
    ret = fs_read(&f, rd_buf, TEST_PAYLOAD_BYTES);
    if (ret < 0) {
        LOG_ERR("fs_read failed (%d)", ret);
        fs_close(&f);
        return ret;
    }
    fs_close(&f);

    if (memcmp(wr_buf, rd_buf, TEST_PAYLOAD_BYTES) != 0) {
        LOG_ERR("Read-back MISMATCH — possible data corruption");
        return -EIO;
    }
    LOG_INF("Read back %d bytes — MATCH", TEST_PAYLOAD_BYTES);

    return 0;
}

/* ------------------------------------------------------------------ */
/*  Entry point                                                        */
/* ------------------------------------------------------------------ */
int main(void)
{
    if (led_init() != 0) {
        /* No LED hardware — log-only mode, continue */
    } else {
        led_white();
    }

    /* Small boot delay so the serial monitor has time to connect
     * before the first log line. 500 ms is enough for CMSIS-DAP. */
    k_msleep(500);

    LOG_INF("=== SD Card Isolated Test (AcoustEEEcare v6.6) ===");

    led_cyan();

    int ret = init_sd_card();
    if (ret != 0) {
        LOG_ERR("init_sd_card FAILED (%d)", ret);
        LOG_ERR("  CS      — overlay: gpio0 pin 3 (D1/P0.03); verify physical wire");
        LOG_ERR("  Format  — must be FAT32; exFAT needs CONFIG_FS_FATFS_EXFAT=y");
        LOG_ERR("  Power   — SD module needs stable 3.3 V");
        LOG_ERR("  SPI     — D8/D9/D10 wired to SCK/MISO/MOSI?");
        led_error_blink();   /* never returns */
    }

    ret = run_payload_test();
    if (ret != 0) {
        LOG_ERR("Payload test FAILED (%d)", ret);
        led_error_blink();
    }

    led_green();

    LOG_INF("=== ALL TESTS PASSED ===");
    LOG_INF("File '%s' left on card — inspect on PC to confirm.", TEST_FILE);

    return 0;
}