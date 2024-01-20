#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main()
{
    char line1[]="abc";
    char line2[]="Abc";
    strcat(line1,line2);
    printf("%c",strcat(line1,line2));
    system("pause");
    return 0;
}
