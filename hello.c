void puts(const char *msg);
void putd(int n);

void f(void) {
    puts("f()");
}

int g(void) {
    return 69;
}

int main(void) {
    f();
    putd(g());
    f();
    return 0;
}
