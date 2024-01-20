#include <stdio.h>
#include <stdlib.h>
void decimalToBinary(int decimal) {
    int binary[32];//定义一个32长度的数组，用于储存每个二进制位数
    int i = 0;
    while (decimal > 0) {
        binary[i] = decimal % 2;
        decimal = decimal / 2;
        i++;
    }
    printf("二进制数为：");
    for (int j = i - 1; j >= 0; j--)//倒序输出储存在binary数组中的每位数字
    {
        printf("%d", binary[j]);
    }
}
int main() {
    int decimal;
    printf("请输入十进制数：");
    scanf("%d",&decimal);
    decimalToBinary(decimal);
    system("pause");
    return 0;
}