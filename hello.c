void puts(const char *msg);
void putd(int n);

int x;
int main(void) {
    puts("Hello, World!");

    x = 42;
    putd(x);
    x = x + 1;
    putd(x);
    return 0;
}
