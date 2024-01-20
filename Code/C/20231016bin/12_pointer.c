// #include <stdio.h>
 
// const int MAX = 3;
 
// int main ()
// {
//    int  var[] = {10, 100, 200};
//    int  i, *ptr;
 
//    /* 指针中的数组地址 */
//    ptr = var;
//    for ( i = 0; i < MAX; i++)
//    {
 
//       printf("存储地址：var[%d] = %p\n", i, ptr );
//       printf("存储值：var[%d] = %d\n", i, *ptr );
//       printf("存储地址:var[%d] = %p\n", i, &var[i] );
//       printf("存储值:var[%d] = %d\n", i, var[i] );//没有区别

//       /* 指向下一个位置 */
//       ptr++;
//    }
//    return 0;
// }
// #include <stdio.h>

// int main() {
//     char *str_array[3] = {"apple", "blueberries", "cherry"};
    
//     for (int i = 0; i < 3; i++) {
//         printf("%s\n", str_array[i]);
//     }
//    getchar();
//     return 0;
// }
// ----------------------------------------------------------------
#include <stdio.h>

// int main() {
//     char str_array[3][13] = {"apple", "blueberries", "cherry"}; // 13是最长字符串"blueberries"的长度+1
//    //  for (int i = 0; i < 3; i++) {
//         printf("%s\n", str_array[1]);
//    //  }
//    getchar();
//     return 0;
// }
//-----------------------------------------------------
// #include <stdio.h>

// int add(int a, int b) { return a + b; }
// int subtract(int a, int b) { return a - b; }

// int main() {
//     int (*func_ptr_array[2])(int, int) = {add, subtract};
//     printf("%d\n", func_ptr_array[0](5, 3)); // Outputs 8
//     printf("%d\n", func_ptr_array[1](5, 3)); // Outputs 2
//     getchar();
//     return 0;
// }
//---------------------------------------------------------------
#include <stdio.h>
 
int main ()
{
   int  V;
   int  *Pt1;
   int  **Pt2;
   int  ***Pt3;
   int 
 
   V = 100;
 
   /* 获取 V 的地址 */
   Pt1 = &V;
 
   /* 使用运算符 & 获取 Pt1 的地址 */
   Pt2 = &Pt1;
   Pt3 = &Pt2;
   /* 使用 pptr 获取值 */
   printf("var = %d\n", V );
   printf("Pt1 = %p\n", Pt1 );
   printf("*Pt1 = %d\n", *Pt1 );
   printf("Pt2 = %p\n", Pt2 );
   printf("*Pt2 = %#x\n", *Pt2);//Pt2的值即为Pt1的地址
   printf("**Pt2 = %d\n", **Pt2);
   printf("***Pt3 = %d\n",***Pt3);
   getchar();
   return 0;
}
