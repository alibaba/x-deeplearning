#include <getopt.h>
#include <easy/easy_io.h>
#include "packet.h"

// command line params
typedef struct cmdline_param {
    // server port
    int port;
    // server io thread count
    int io_thread_cnt;
} cmdline_param;

// --------------------------------------------------

static void print_usage(char *prog_name)
{
    fprintf(stderr, "%s -p port [-t thread_cnt]\n"
            "    -p, --port              server port\n"
            "    -t, --io_thread_cnt     thread count for listen, default: 1\n"
            "    -h, --help              display this help and exit\n"
            "eg: %s -p 5000\n\n", prog_name, prog_name);
}

static int parse_cmd_line(int argc, char *const argv[], cmdline_param *cp)
{
    int                     opt;
    const char              *opt_string = "hVp:t:";
    struct option           long_opts[] = {
        {"port", 1, NULL, 'p'},
        {"io_thread_cnt", 1, NULL, 't'},
        {"help", 0, NULL, 'h'},
        {0, 0, 0, 0}
    };

    opterr = 0;

    while ((opt = getopt_long(argc, argv, opt_string, long_opts, NULL)) != -1) {
        switch (opt) {
        case 'p':
            cp->port = atoi(optarg);
            break;

        case 't':
            cp->io_thread_cnt = atoi(optarg);
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
    // use ipacket as the opacket
    r->opacket = r->ipacket;
    return EASY_OK;
}

// connect func, will be called by framework
static int echo_connect(easy_connection_t *c)
{
    char                    *str = "hello, world!\n";
    easy_error_log("new connection:%p at %d\n", c, c->fd);
    easy_ignore(write(c->fd, str, strlen(str) + 1));
    return EASY_OK;
}

// main
int main(int argc, char **argv)
{
    cmdline_param cp;
    easy_listen_t *listen;
    easy_io_handler_pt io_handler;
    int ret;

    // default params
    memset(&cp, 0, sizeof(cmdline_param));
    cp.io_thread_cnt = 1;

    // parse cmd line
    if (parse_cmd_line(argc, argv, &cp) == EASY_ERROR)
        return EASY_ERROR;

    if (cp.port == 0) {
        print_usage(argv[0]);
        return EASY_ERROR;
    }

    // create all io threads
    if (!easy_io_create(cp.io_thread_cnt)) {
        easy_error_log("easy_io_init error.\n");
        return EASY_ERROR;
    }

    // set defer accept = false
    // so we will create the conn immediately when recv the ack from client
    easy_io_var.tcp_defer_accept = 0;
    easy_io_var.no_redispatch = EASY_FIRST_DISPATCH;

    // set handlers
    memset(&io_handler, 0, sizeof(io_handler));
    // decode function
    io_handler.decode = echo_decode;
    // encode function
    io_handler.encode = echo_encode;
    // process function
    io_handler.process = echo_process;
    // connect function
    io_handler.on_connect = echo_connect;

    // start listen
    if ((listen = easy_io_add_listen(NULL, cp.port, &io_handler)) == NULL) {
        easy_error_log("easy_io_add_listen error, port: %d, %s\n",
                       cp.port, strerror(errno));
        return EASY_ERROR;
    } else {
        easy_error_log("listen start, port = %d\n", cp.port);
    }

    // watch stat
    ev_timer                stat_watcher;
    easy_io_stat_t          iostat;
    easy_io_stat_watcher_start(&stat_watcher, 5.0, &iostat, NULL);

    // easy start ...
    if (easy_io_start()) {
        easy_error_log("easy_io_start error.\n");
        return EASY_ERROR;
    }

    // wait here
    ret = easy_io_wait();
    easy_io_destroy();

    return ret;
}

