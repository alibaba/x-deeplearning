#include <stdint.h>
#include <getopt.h>
#include <easy/easy_io.h>
#include <iostream>
#include <sys/time.h>
#include "packet.h"

// command line params
typedef struct cmdline_param {
    // server address
    easy_addr_t             address;
    // io thread count
    int                     io_thread_cnt;
    // the packet size of one request
    int                     request_size;
    // request count
    int64_t                 request_cnt;
} cmdline_param;

// global param
cmdline_param cp;

// save rtts, max count 10000
int64_t rtts[10000];

// send packet count && recv response count
int64_t send_cnt = 0;
int64_t process_cnt = 0;

// record start and end time
uint64_t start_time = 0;
uint64_t end_time = 0;

// ------------------------------------------------------

static void print_usage(char *prog_name) 
{
    fprintf(stderr, "%s -H host:port -c conn_cnt -n req_cnt -s size [-t thread_cnt]\n"
            "    -H, --host              server address\n"
            "    -n, --req_cnt           request count\n"
            "    -s, --req_size          packet size of every request\n"
            "    -t, --io_thread_cnt     thread count for listen, default: 1\n"
            "    -h, --help              display this help and exit\n"
            "eg: %s -Hlocalhost:5000 -n1000 -s512\n\n", prog_name, prog_name);
}


static int parse_cmd_line(int argc, char *const argv[], cmdline_param *cp)
{
    int opt;
    const char *opt_string = "hVH:c:n:s:t:i:b:";
    struct option long_opts[] = {
        {"host", 1, NULL, 'H'},
        {"req_cnt", 1, NULL, 'n'},
        {"req_size", 1, NULL, 's'},
        {"io_thread_cnt", 1, NULL, 't'},
        {"help", 0, NULL, 'h'},
        {0, 0, 0, 0}
    };

    opterr = 0;

    while ((opt = getopt_long(argc, argv, opt_string, long_opts, NULL)) != -1) {
        switch (opt) {
        case 'H':
            cp->address = easy_inet_str_to_addr(optarg, 0);
            break;

        case 't':
            cp->io_thread_cnt = atoi(optarg);
            break;

        case 'n':
            cp->request_cnt = atoi(optarg);
            break;

        case 's':
            if ((cp->request_size = atoi(optarg)) <= 0)
                cp->request_size = 128;
            break;

        case 'h':
            print_usage(argv[0]);
            return EASY_ERROR;

        default:
            break;
        }
    }

    return EASY_OK;
}

// process func, will be called by framework
static int echo_process(easy_request_t *r)
{
    echo_packet_t           *reply;

    reply = (echo_packet_t *) r->ipacket;

    // save a rtt
    rtts[process_cnt] = (int64_t)(((easy_session_t *)r->ms)->now * 1000000);

    easy_session_destroy(r->ms);

    if (!reply)
        return EASY_ERROR;

    ++process_cnt;

    // receive all response
    if (process_cnt >= cp.request_cnt) {
        easy_io_stop();
    }

    return EASY_OK;
}


// send packet func, will be called by framework
static int echo_new_packet(easy_connection_t *c)
{
    easy_session_t          *s;
    echo_packet_t           *packet;

    // the first packet
    if (send_cnt == 0) {
        // start time
        struct timeval tv;
        gettimeofday(&tv,NULL);
        start_time = tv.tv_sec * 1000000 + tv.tv_usec;
        std::cout << "start time: " << start_time << std::endl;
    }

    // send finish
    if (send_cnt >= cp.request_cnt) {
        struct timeval tv;
        gettimeofday(&tv,NULL);
        start_time = tv.tv_sec * 1000000 + tv.tv_usec;
        std::cout << "end time: " << start_time << ", send_cnt = " << send_cnt<< std::endl;

        return EASY_OK;
    }

    ++send_cnt;

    // create a buf
    if ((packet = easy_session_packet_create(echo_packet_t, s, cp.request_size)) == NULL)
        return EASY_ERROR;

    packet->data = &packet->buffer[0];
    packet->len = cp.request_size;
    //memset(packet->data, 0, packet->len);
    //easy_atomic_add(&cp.send_byte, cp.request_size);

    easy_connection_send_session(c, s);
    return EASY_OK;
}

// disconnect func
static int echo_disconnect(easy_connection_t *c)
{
    easy_error_log("connect disconnect.\n");
    easy_io_stop();
    return EASY_OK;
}


// main
int main(int argc, char **argv)
{
    int i, ret;
    easy_io_handler_pt io_handler;

    // default
    memset(&cp, 0, sizeof(cmdline_param));
    cp.io_thread_cnt = 1;
    cp.request_cnt = 99999;
    cp.request_size = 128;

    // parse cmd line
    if (parse_cmd_line(argc, argv, &cp) == EASY_ERROR)
        return EASY_ERROR;

    if (cp.address.family == 0) {
        print_usage(argv[0]);
        return EASY_ERROR;
    }

    // set affinity
    easy_io_var.affinity_enable = 1;

    // create io threads
    if (!easy_io_create(cp.io_thread_cnt)) {
        easy_error_log("easy_io_init error.\n");
        return EASY_ERROR;
    }

    // set handlers
    memset(&io_handler, 0, sizeof(io_handler));
    // decode function
    io_handler.decode = echo_decode;
    // encode function
    io_handler.encode = echo_encode;
    // process function
    io_handler.process = echo_process;
    // send new packet function
    io_handler.new_packet = echo_new_packet;
    // disconnect function
    io_handler.on_disconnect = echo_disconnect;
    // user define data
    io_handler.user_data = (void *)(long)cp.request_size;

    easy_addr_t addr;
    memcpy(&addr, &cp.address, sizeof(easy_addr_t));
    addr.cidx = 0;

    if (easy_io_connect(addr, &io_handler, 0, NULL) != EASY_OK) {
        char buffer[32];
        easy_error_log("connection failure: %s\n", easy_inet_addr_to_str(&cp.address, buffer, 32));
    }
    
    // easy io start
    if (easy_io_start()) {
        easy_error_log("easy_io_start error.\n");
        return EASY_ERROR;
    }

    // easy wait here
    ret = easy_io_wait();

    // end time ...
    struct timeval tv;
    gettimeofday(&tv,NULL);
    end_time = tv.tv_sec * 1000000 + tv.tv_usec;
    std::cout << "end time: " << end_time << std::endl;
    std::cout << "Cost time: " << end_time - start_time << std::endl;

    // print rtts
    int64_t total = 0;
    for (i = 0; i < cp.request_cnt; ++i) {
        total += rtts[i];
        std::cout << rtts[i] << std::endl;
    }
    std::cout << "Avg rtt = " << total / cp.request_cnt << std::endl;

    // destroy io context
    easy_io_destroy();

    return ret;
}


