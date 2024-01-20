// #include <stdio.h>
// #include <stdlib.h>
// struct Books
// {
//     char title[50];
//     char author[50];
//     char subject[100];
//     int book_id;
// }book = {"Ctest1","Harichane","编程语言",123465};//定义结构变量
// /*定义结构体*/
// int main()
// {   
//     struct Books book1 ={"你好","mememe","idontknow",114514};
//     printf("title: %s\nauthor = %s\nsubject = %s\nbook_id:%d\n",book.title,book.author,book.subject,book.book_id);
//     printf("----------------------------------\n");
//     printf("title: %s\nauthor = %s\nsubject = %s\nbook_id:%d\n",book1.title,book1.author,book1.subject,book1.book_id);
//     getchar();
//     return 0;
// }
#include <stdio.h>
#include <string.h>

struct Books {
    char title[50];
    char author[50];
    char subject[100];
    int book_id;
};

/* 函数声明 */
void printBook(struct Books);

int main() {
    struct Books Book1;        /* 声明 Book1，类型为 Books */
    struct Books Book2;        /* 声明 Book2，类型为 Books */

    /* Book1 详述 */
    strcpy(Book1.title, "C Programming");
    strcpy(Book1.author, "Nuha Ali"); 
    strcpy(Book1.subject, "C Programming Tutorial");
    Book1.book_id = 6495407;

    /* Book2 详述 */
    strcpy(Book2.title, "Telecom Billing");
    strcpy(Book2.author, "Zara Ali");
    strcpy(Book2.subject, "Telecom Billing Tutorial");
    Book2.book_id = 6495700;

    /* 输出 Book1 信息 */
    printBook(Book1);

    /* 输出 Book2 信息 */
    printBook(Book2);
    printf("%p\n",&Book1);
    printf("size of struct = %d",sizeof(Book1));
    getchar();
    return 0;
}

void printBook(struct Books book) {
    printf("Book title : %s\n", book.title);
    printf("Book author : %s\n", book.author);
    printf("Book subject : %s\n", book.subject);
    printf("Book book_id : %d\n", book.book_id);
}
