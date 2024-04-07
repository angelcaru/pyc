void puts(const char *msg);
void putd(int n);

char *msg;
char b;
char zero;
int main(void) {
    msg = "hello";
    zero = 0;

    while (*msg) {
        b = *msg;
        puts(&b);
        msg = msg + 1;
    }

    return 0;
}
