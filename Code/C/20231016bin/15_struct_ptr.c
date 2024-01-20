    #include <stdio.h>
    #include <stdlib.h>
    #include <string.h>
    struct Books
    {
        char title[50];
        char author[50];
        char subject[100];
        int book_id;
    };//定义结构体
    int main()
    {
        struct Books Book1;//定义结构体变量Book1
        struct Books *struct_ptr = &Book1;//定义指向结构体变量Book1的 结构体指针：struct_ptr
        strcpy(Book1.author,"Harichane");
        printf("%s\n",struct_ptr->author);//访问结构体的成员：->
        printf("%zu\n",sizeof(Book1));
        system("pause");
        return 0;
    }