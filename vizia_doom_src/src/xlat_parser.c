/* Driver template for the LEMON parser generator.
** The author disclaims copyright to this source code.
*/
/* First off, code is included which follows the "include" declaration
** in the input file. */
#include <stdio.h>
#include <string.h>
#include <assert.h>

#ifdef _MSC_VER
#define CDECL __cdecl
#else
#define CDECL
#endif

/* Next is all token values, in a form suitable for use by makeheaders.
** This section will be null unless lemon is run with the -m switch.
*/
/* 
** These constants (all generated automatically by the parser generator)
** specify the various kinds of tokens (terminals) that the parser
** understands. 
**
** Each symbol here is a terminal symbol in the grammar.
*/
/* Make sure the INTERFACE macro is defined.
*/
#ifndef INTERFACE
# define INTERFACE 1
#endif
/* The next thing included is series of defines which control
** various aspects of the generated parser.
**    YYCODETYPE         is the data type used for storing terminal
**                       and nonterminal numbers.  "unsigned char" is
**                       used if there are fewer than 250 terminals
**                       and nonterminals.  "int" is used otherwise.
**    YYNOCODE           is a number of type YYCODETYPE which corresponds
**                       to no legal terminal or nonterminal number.  This
**                       number is used to fill in empty slots of the hash 
**                       table.
**    YYFALLBACK         If defined, this indicates that one or more tokens
**                       have fall-back values which should be used if the
**                       original value of the token will not parse.
**    YYACTIONTYPE       is the data type used for storing terminal
**                       and nonterminal numbers.  "unsigned char" is
**                       used if there are fewer than 250 rules and
**                       states combined.  "int" is used otherwise.
**    XlatParseTOKENTYPE     is the data type used for minor tokens given 
**                       directly to the parser from the tokenizer.
**    YYMINORTYPE        is the data type used for all minor tokens.
**                       This is typically a union of many types, one of
**                       which is XlatParseTOKENTYPE.  The entry in the union
**                       for base tokens is called "yy0".
**    YYSTACKDEPTH       is the maximum depth of the parser's stack.  If
**                       zero the stack is dynamically sized using realloc()
**    XlatParseARG_SDECL     A static variable declaration for the %extra_argument
**    XlatParseARG_PDECL     A parameter declaration for the %extra_argument
**    XlatParseARG_STORE     Code to store %extra_argument into yypParser
**    XlatParseARG_FETCH     Code to extract %extra_argument from yypParser
**    YYNSTATE           the combined number of states.
**    YYNRULE            the number of rules in the grammar
**    YYERRORSYMBOL      is the code number of the error symbol.  If not
**                       defined, then do no error processing.
*/
#define YYCODETYPE unsigned char
#define YYNOCODE 71
#define YYACTIONTYPE unsigned char
#define XlatParseTOKENTYPE FParseToken
typedef union {
  XlatParseTOKENTYPE yy0;
  SpecialArgs yy7;
  MoreFilters * yy8;
  int yy32;
  ParseBoomArg yy63;
  ListFilter yy83;
  SpecialArg yy120;
  MoreLines * yy129;
  FBoomArg yy130;
} YYMINORTYPE;
#ifndef YYSTACKDEPTH
#define YYSTACKDEPTH 100
#endif
#define XlatParseARG_SDECL  FParseContext *context ;
#define XlatParseARG_PDECL , FParseContext *context 
#define XlatParseARG_FETCH  FParseContext *context  = yypParser->context 
#define XlatParseARG_STORE yypParser->context  = context 
#define YYNSTATE 155
#define YYNRULE 78
#define YY_NO_ACTION      (YYNSTATE+YYNRULE+2)
#define YY_ACCEPT_ACTION  (YYNSTATE+YYNRULE+1)
#define YY_ERROR_ACTION   (YYNSTATE+YYNRULE)

/* Next are that tables used to determine what action to take based on the
** current state and lookahead token.  These tables are used to implement
** functions that take a state number and lookahead value and return an
** action integer.  
**
** Suppose the action integer is N.  Then the action is determined as
** follows
**
**   0 <= N < YYNSTATE                  Shift N.  That is, push the lookahead
**                                      token onto the stack and goto state N.
**
**   YYNSTATE <= N < YYNSTATE+YYNRULE   Reduce by rule N-YYNSTATE.
**
**   N == YYNSTATE+YYNRULE              A syntax error has occurred.
**
**   N == YYNSTATE+YYNRULE+1            The parser accepts its input.
**
**   N == YYNSTATE+YYNRULE+2            No such action.  Denotes unused
**                                      slots in the yy_action[] table.
**
** The action table is constructed as a single large table named yy_action[].
** Given state S and lookahead X, the action is computed as
**
**      yy_action[ yy_shift_ofst[S] + X ]
**
** If the index value yy_shift_ofst[S]+X is out of range or if the value
** yy_lookahead[yy_shift_ofst[S]+X] is not equal to X or if yy_shift_ofst[S]
** is equal to YY_SHIFT_USE_DFLT, it means that the action is not in the table
** and that yy_default[S] should be used instead.  
**
** The formula above is for computing the action when the lookahead is
** a terminal symbol.  If the lookahead is a non-terminal (as occurs after
** a reduce action) then the yy_reduce_ofst[] array is used in place of
** the yy_shift_ofst[] array and YY_REDUCE_USE_DFLT is used in place of
** YY_SHIFT_USE_DFLT.
**
** The following are the tables generated in this section:
**
**  yy_action[]        A single table containing all actions.
**  yy_lookahead[]     A table containing the lookahead for each entry in
**                     yy_action.  Used to detect hash collisions.
**  yy_shift_ofst[]    For each state, the offset into yy_action for
**                     shifting terminals.
**  yy_reduce_ofst[]   For each state, the offset into yy_action for
**                     shifting non-terminals after a reduce.
**  yy_default[]       Default action for each state.
*/
static const YYACTIONTYPE yy_action[] = {
 /*     0 */   100,  101,  102,  103,  104,  105,  106,  107,  108,   66,
 /*    10 */    10,  234,    1,   86,   68,  130,   30,   32,   31,   26,
 /*    20 */    25,   27,   28,   29,   84,   30,   32,   31,   26,   25,
 /*    30 */    27,   28,   29,   30,   32,   31,   26,   25,   27,   28,
 /*    40 */    29,   33,   81,  119,  115,   59,  149,  110,   34,   44,
 /*    50 */    99,  151,  152,  155,  109,  145,  136,   97,   33,  139,
 /*    60 */   140,  141,  142,  143,  110,   34,   21,   82,   54,   79,
 /*    70 */    85,   43,  122,   22,  137,   38,   26,   25,   27,   28,
 /*    80 */    29,   62,  123,  138,   96,   80,   23,   87,   68,  129,
 /*    90 */    88,  117,   45,   50,   49,   30,   32,   31,   26,   25,
 /*   100 */    27,   28,   29,  153,  124,  125,  116,  126,   71,   30,
 /*   110 */    32,   31,   26,   25,   27,   28,   29,   30,   32,   31,
 /*   120 */    26,   25,   27,   28,   29,   76,   12,    2,   73,   18,
 /*   130 */    20,   19,   14,   13,   15,   16,   17,   48,    5,   93,
 /*   140 */   128,  127,   55,   30,   32,   31,   26,   25,   27,   28,
 /*   150 */    29,   30,   32,   31,   26,   25,   27,   28,   29,   26,
 /*   160 */    25,   27,   28,   29,   91,   48,    5,  133,   40,   77,
 /*   170 */    60,   30,   32,   31,   26,   25,   27,   28,   29,   95,
 /*   180 */    94,   78,   92,  111,  112,   30,   32,   31,   26,   25,
 /*   190 */    27,   28,   29,   30,   32,   31,   26,   25,   27,   28,
 /*   200 */    29,   90,   68,  129,   88,    6,    9,  113,   30,   32,
 /*   210 */    31,   26,   25,   27,   28,   29,   14,   13,   15,   16,
 /*   220 */    17,   41,   70,   75,   30,   32,   31,   26,   25,   27,
 /*   230 */    28,   29,   31,   26,   25,   27,   28,   29,  144,   27,
 /*   240 */    28,   29,   24,  118,   72,   30,   32,   31,   26,   25,
 /*   250 */    27,   28,   29,   30,   32,   46,   26,   25,   27,   28,
 /*   260 */    29,   30,   32,   31,   26,   25,   27,   28,   29,  114,
 /*   270 */    33,   47,   51,   52,   35,  148,  110,   34,   30,   32,
 /*   280 */    31,   26,   25,   27,   28,   29,   15,   16,   17,   67,
 /*   290 */    53,  154,   56,   57,   18,   20,   19,   14,   13,   15,
 /*   300 */    16,   17,   32,   31,   26,   25,   27,   28,   29,   20,
 /*   310 */    19,   14,   13,   15,   16,   17,   19,   14,   13,   15,
 /*   320 */    16,   17,   33,   58,   33,   69,   61,   36,  110,   34,
 /*   330 */   110,   34,   89,   60,   98,   11,   63,   64,   74,   65,
 /*   340 */    83,  120,   95,  135,   37,  121,    8,    3,    4,  131,
 /*   350 */    39,  134,    7,  132,   42,  235,  235,  235,  146,  235,
 /*   360 */   235,  147,  235,  150,
};
static const YYCODETYPE yy_lookahead[] = {
 /*     0 */    43,   44,   45,   46,   47,   48,   49,   50,   51,   52,
 /*    10 */    53,   41,   42,   56,   58,   59,    2,    3,    4,    5,
 /*    20 */     6,    7,    8,    9,   15,    2,    3,    4,    5,    6,
 /*    30 */     7,    8,    9,    2,    3,    4,    5,    6,    7,    8,
 /*    40 */     9,    5,   54,   55,   13,   52,   32,   11,   12,   68,
 /*    50 */    36,   37,   38,    0,    1,   32,   63,   34,    5,   24,
 /*    60 */    25,   26,   27,   28,   11,   12,    5,   14,   52,   58,
 /*    70 */    17,   35,   11,   12,   20,   22,    5,    6,    7,    8,
 /*    80 */     9,   52,   21,   29,   31,   58,   33,   57,   58,   59,
 /*    90 */    60,   16,   39,   52,   19,    2,    3,    4,    5,    6,
 /*   100 */     7,    8,    9,   32,   58,   58,   13,   58,   58,    2,
 /*   110 */     3,    4,    5,    6,    7,    8,    9,    2,    3,    4,
 /*   120 */     5,    6,    7,    8,    9,   58,   19,   12,   58,    2,
 /*   130 */     3,    4,    5,    6,    7,    8,    9,   65,   66,   67,
 /*   140 */    13,   58,   58,    2,    3,    4,    5,    6,    7,    8,
 /*   150 */     9,    2,    3,    4,    5,    6,    7,    8,    9,    5,
 /*   160 */     6,    7,    8,    9,   23,   65,   66,   67,   19,   52,
 /*   170 */    52,    2,    3,    4,    5,    6,    7,    8,    9,   61,
 /*   180 */    62,   52,   13,   52,   52,    2,    3,    4,    5,    6,
 /*   190 */     7,    8,    9,    2,    3,    4,    5,    6,    7,    8,
 /*   200 */     9,   57,   58,   59,   60,   22,   64,   52,    2,    3,
 /*   210 */     4,    5,    6,    7,    8,    9,    5,    6,    7,    8,
 /*   220 */     9,   30,   52,   52,    2,    3,    4,    5,    6,    7,
 /*   230 */     8,    9,    4,    5,    6,    7,    8,    9,   32,    7,
 /*   240 */     8,    9,   20,   55,   52,    2,    3,    4,    5,    6,
 /*   250 */     7,    8,    9,    2,    3,    4,    5,    6,    7,    8,
 /*   260 */     9,    2,    3,    4,    5,    6,    7,    8,    9,   52,
 /*   270 */     5,   20,   52,   52,   12,   32,   11,   12,    2,    3,
 /*   280 */     4,    5,    6,    7,    8,    9,    7,    8,    9,   52,
 /*   290 */    52,   32,   52,   52,    2,    3,    4,    5,    6,    7,
 /*   300 */     8,    9,    3,    4,    5,    6,    7,    8,    9,    3,
 /*   310 */     4,    5,    6,    7,    8,    9,    4,    5,    6,    7,
 /*   320 */     8,    9,    5,   52,    5,   52,   52,   20,   11,   12,
 /*   330 */    11,   12,   15,   52,   15,   52,   52,   52,   52,   52,
 /*   340 */    15,   18,   61,   62,   20,   13,   19,   12,   18,   13,
 /*   350 */    12,   23,   19,   16,   20,   70,   70,   70,   32,   70,
 /*   360 */    70,   32,   70,   32,
};
#define YY_SHIFT_USE_DFLT (-1)
#define YY_SHIFT_MAX 99
static const short yy_shift_ofst[] = {
 /*     0 */    -1,   53,   61,   61,   35,   35,  265,  265,   61,  265,
 /*    10 */     9,   14,  317,   61,   61,   61,   61,   61,   61,   61,
 /*    20 */    61,   61,   61,   36,  319,  265,  265,  265,  265,  265,
 /*    30 */   265,  265,  265,  265,  265,  265,  265,  265,  265,  265,
 /*    40 */   265,  265,  265,  265,  265,  265,  265,  265,   54,    9,
 /*    50 */    23,   31,   93,  107,  115,  127,  141,  149,  169,  183,
 /*    60 */   191,  206,  222,  243,  251,  259,  276,  276,  292,  276,
 /*    70 */   299,  306,  228,  312,   71,  154,  211,  232,  232,  279,
 /*    80 */   279,   75,  325,  262,  307,  323,  324,  332,  327,  335,
 /*    90 */   336,  338,  330,  337,  328,  333,  334,  326,  329,  331,
};
#define YY_REDUCE_USE_DFLT (-45)
#define YY_REDUCE_MAX 49
static const short yy_reduce_ofst[] = {
 /*     0 */   -30,  -43,   30,  144,   72,  100,  118,  281,  -44,   -7,
 /*    10 */   -12,  -19,   16,   11,   27,   46,   47,   49,   50,   67,
 /*    20 */    70,   83,   84,   29,   41,  117,  129,  131,  132,  155,
 /*    30 */   170,  171,  192,  217,  220,  221,  237,  238,  240,  241,
 /*    40 */   271,  273,  274,  283,  284,  285,  286,  287,  142,  188,
};
static const YYACTIONTYPE yy_default[] = {
 /*     0 */   156,  233,  204,  204,  207,  207,  233,  233,  233,  233,
 /*    10 */   181,  233,  233,  233,  233,  233,  233,  233,  233,  233,
 /*    20 */   233,  233,  233,  233,  233,  233,  233,  233,  233,  233,
 /*    30 */   233,  233,  233,  233,  233,  233,  233,  233,  233,  233,
 /*    40 */   233,  233,  233,  233,  233,  233,  233,  233,  233,  233,
 /*    50 */   233,  233,  233,  233,  233,  233,  233,  233,  233,  217,
 /*    60 */   233,  233,  233,  233,  233,  233,  186,  185,  201,  221,
 /*    70 */   173,  196,  175,  198,  174,  174,  197,  168,  169,  191,
 /*    80 */   192,  233,  233,  233,  184,  233,  233,  233,  205,  233,
 /*    90 */   233,  233,  233,  233,  233,  219,  233,  233,  233,  233,
 /*   100 */   157,  158,  159,  160,  161,  162,  163,  164,  165,  166,
 /*   110 */   167,  170,  171,  172,  176,  177,  178,  179,  183,  182,
 /*   120 */   180,  187,  189,  190,  193,  194,  195,  199,  200,  202,
 /*   130 */   203,  188,  206,  208,  218,  220,  209,  215,  216,  210,
 /*   140 */   211,  212,  213,  214,  222,  223,  225,  224,  226,  227,
 /*   150 */   228,  229,  230,  232,  231,
};
#define YY_SZ_ACTTAB (int)(sizeof(yy_action)/sizeof(yy_action[0]))

/* The next table maps tokens into fallback tokens.  If a construct
** like the following:
** 
**      %fallback ID X Y Z.
**
** appears in the grammer, then ID becomes a fallback token for X, Y,
** and Z.  Whenever one of the tokens X, Y, or Z is input to the parser
** but it does not parse, the type of the token is changed to ID and
** the parse is retried before an error is thrown.
*/
#ifdef YYFALLBACK
static const YYCODETYPE yyFallback[] = {
};
#endif /* YYFALLBACK */

/* The following structure represents a single element of the
** parser's stack.  Information stored includes:
**
**   +  The state number for the parser at this level of the stack.
**
**   +  The value of the token stored at this level of the stack.
**      (In other words, the "major" token.)
**
**   +  The semantic value stored at this level of the stack.  This is
**      the information used by the action routines in the grammar.
**      It is sometimes called the "minor" token.
*/
struct yyStackEntry {
  int stateno;       /* The state-number */
  int major;         /* The major token value.  This is the code
                     ** number for the token at this stack level */
  YYMINORTYPE minor; /* The user-supplied minor token value.  This
                     ** is the value of the token  */
};
typedef struct yyStackEntry yyStackEntry;

/* The state of the parser is completely contained in an instance of
** the following structure */
struct yyParser {
  int yyidx;                    /* Index of top element in stack */
  int yyerrcnt;                 /* Shifts left before out of the error */
  XlatParseARG_SDECL                /* A place to hold %extra_argument */
#if YYSTACKDEPTH<=0
  int yystksz;                  /* Current side of the stack */
  yyStackEntry *yystack;        /* The parser's stack */
#else
  yyStackEntry yystack[YYSTACKDEPTH];  /* The parser's stack */
#endif
};
typedef struct yyParser yyParser;

#ifndef NDEBUG
#include <stdio.h>
static FILE *yyTraceFILE = 0;
static char *yyTracePrompt = 0;
#endif /* NDEBUG */

#ifndef NDEBUG
/* 
** Turn parser tracing on by giving a stream to which to write the trace
** and a prompt to preface each trace message.  Tracing is turned off
** by making either argument NULL 
**
** Inputs:
** <ul>
** <li> A FILE* to which trace output should be written.
**      If NULL, then tracing is turned off.
** <li> A prefix string written at the beginning of every
**      line of trace output.  If NULL, then tracing is
**      turned off.
** </ul>
**
** Outputs:
** None.
*/
void XlatParseTrace(FILE *TraceFILE, char *zTracePrompt){
  yyTraceFILE = TraceFILE;
  yyTracePrompt = zTracePrompt;
  if( yyTraceFILE==0 ) yyTracePrompt = 0;
  else if( yyTracePrompt==0 ) yyTraceFILE = 0;
}
#endif /* NDEBUG */

#ifndef NDEBUG
/* For tracing shifts, the names of all terminals and nonterminals
** are required.  The following table supplies these names */
static const char *const yyTokenName[] = { 
  "$",             "NOP",           "OR",            "XOR",         
  "AND",           "MINUS",         "PLUS",          "MULTIPLY",    
  "DIVIDE",        "MODULUS",       "NEG",           "NUM",         
  "LPAREN",        "RPAREN",        "DEFINE",        "SYM",         
  "RBRACE",        "ENUM",          "LBRACE",        "COMMA",       
  "EQUALS",        "TAG",           "LBRACKET",      "RBRACKET",    
  "FLAGS",         "ARG2",          "ARG3",          "ARG4",        
  "ARG5",          "OR_EQUAL",      "COLON",         "MAXLINESPECIAL",
  "SEMICOLON",     "SECTOR",        "NOBITMASK",     "BITMASK",     
  "CLEAR",         "LSHASSIGN",     "RSHASSIGN",     "LINEFLAG",    
  "error",         "main",          "translation_unit",  "external_declaration",
  "define_statement",  "enum_statement",  "linetype_declaration",  "boom_declaration",
  "sector_declaration",  "lineflag_declaration",  "sector_bitmask",  "maxlinespecial_def",
  "exp",           "enum_open",     "enum_list",     "single_enum", 
  "linetype_exp",  "special_args",  "exp_with_tag",  "special_arg", 
  "multi_special_arg",  "list_val",      "arg_list",      "boom_args",   
  "boom_op",       "boom_selector",  "boom_line",     "boom_body",   
  "sector_op",     "lineflag_op", 
};
#endif /* NDEBUG */

#ifndef NDEBUG
/* For tracing reduce actions, the names of all rules are required.
*/
static const char *const yyRuleName[] = {
 /*   0 */ "main ::= translation_unit",
 /*   1 */ "translation_unit ::=",
 /*   2 */ "translation_unit ::= translation_unit external_declaration",
 /*   3 */ "external_declaration ::= define_statement",
 /*   4 */ "external_declaration ::= enum_statement",
 /*   5 */ "external_declaration ::= linetype_declaration",
 /*   6 */ "external_declaration ::= boom_declaration",
 /*   7 */ "external_declaration ::= sector_declaration",
 /*   8 */ "external_declaration ::= lineflag_declaration",
 /*   9 */ "external_declaration ::= sector_bitmask",
 /*  10 */ "external_declaration ::= maxlinespecial_def",
 /*  11 */ "external_declaration ::= NOP",
 /*  12 */ "exp ::= NUM",
 /*  13 */ "exp ::= exp PLUS exp",
 /*  14 */ "exp ::= exp MINUS exp",
 /*  15 */ "exp ::= exp MULTIPLY exp",
 /*  16 */ "exp ::= exp DIVIDE exp",
 /*  17 */ "exp ::= exp MODULUS exp",
 /*  18 */ "exp ::= exp OR exp",
 /*  19 */ "exp ::= exp AND exp",
 /*  20 */ "exp ::= exp XOR exp",
 /*  21 */ "exp ::= MINUS exp",
 /*  22 */ "exp ::= LPAREN exp RPAREN",
 /*  23 */ "define_statement ::= DEFINE SYM LPAREN exp RPAREN",
 /*  24 */ "enum_statement ::= enum_open enum_list RBRACE",
 /*  25 */ "enum_open ::= ENUM LBRACE",
 /*  26 */ "enum_list ::=",
 /*  27 */ "enum_list ::= single_enum",
 /*  28 */ "enum_list ::= enum_list COMMA single_enum",
 /*  29 */ "single_enum ::= SYM",
 /*  30 */ "single_enum ::= SYM EQUALS exp",
 /*  31 */ "linetype_exp ::= exp",
 /*  32 */ "linetype_declaration ::= linetype_exp EQUALS exp COMMA exp LPAREN special_args RPAREN",
 /*  33 */ "linetype_declaration ::= linetype_exp EQUALS exp COMMA SYM LPAREN special_args RPAREN",
 /*  34 */ "exp_with_tag ::= NUM",
 /*  35 */ "exp_with_tag ::= TAG",
 /*  36 */ "exp_with_tag ::= exp_with_tag PLUS exp_with_tag",
 /*  37 */ "exp_with_tag ::= exp_with_tag MINUS exp_with_tag",
 /*  38 */ "exp_with_tag ::= exp_with_tag MULTIPLY exp_with_tag",
 /*  39 */ "exp_with_tag ::= exp_with_tag DIVIDE exp_with_tag",
 /*  40 */ "exp_with_tag ::= exp_with_tag MODULUS exp_with_tag",
 /*  41 */ "exp_with_tag ::= exp_with_tag OR exp_with_tag",
 /*  42 */ "exp_with_tag ::= exp_with_tag AND exp_with_tag",
 /*  43 */ "exp_with_tag ::= exp_with_tag XOR exp_with_tag",
 /*  44 */ "exp_with_tag ::= MINUS exp_with_tag",
 /*  45 */ "exp_with_tag ::= LPAREN exp_with_tag RPAREN",
 /*  46 */ "special_arg ::= exp_with_tag",
 /*  47 */ "multi_special_arg ::= special_arg",
 /*  48 */ "multi_special_arg ::= multi_special_arg COMMA special_arg",
 /*  49 */ "special_args ::=",
 /*  50 */ "special_args ::= multi_special_arg",
 /*  51 */ "boom_declaration ::= LBRACKET exp RBRACKET LPAREN exp COMMA exp RPAREN LBRACE boom_body RBRACE",
 /*  52 */ "boom_body ::=",
 /*  53 */ "boom_body ::= boom_line boom_body",
 /*  54 */ "boom_line ::= boom_selector boom_op boom_args",
 /*  55 */ "boom_selector ::= FLAGS",
 /*  56 */ "boom_selector ::= ARG2",
 /*  57 */ "boom_selector ::= ARG3",
 /*  58 */ "boom_selector ::= ARG4",
 /*  59 */ "boom_selector ::= ARG5",
 /*  60 */ "boom_op ::= EQUALS",
 /*  61 */ "boom_op ::= OR_EQUAL",
 /*  62 */ "boom_args ::= exp",
 /*  63 */ "boom_args ::= exp LBRACKET arg_list RBRACKET",
 /*  64 */ "arg_list ::= list_val",
 /*  65 */ "arg_list ::= list_val COMMA arg_list",
 /*  66 */ "list_val ::= exp COLON exp",
 /*  67 */ "maxlinespecial_def ::= MAXLINESPECIAL EQUALS exp SEMICOLON",
 /*  68 */ "sector_declaration ::= SECTOR exp EQUALS exp SEMICOLON",
 /*  69 */ "sector_declaration ::= SECTOR exp EQUALS SYM SEMICOLON",
 /*  70 */ "sector_declaration ::= SECTOR exp EQUALS exp NOBITMASK SEMICOLON",
 /*  71 */ "sector_bitmask ::= SECTOR BITMASK exp sector_op exp SEMICOLON",
 /*  72 */ "sector_bitmask ::= SECTOR BITMASK exp SEMICOLON",
 /*  73 */ "sector_bitmask ::= SECTOR BITMASK exp CLEAR SEMICOLON",
 /*  74 */ "sector_op ::= LSHASSIGN",
 /*  75 */ "sector_op ::= RSHASSIGN",
 /*  76 */ "lineflag_declaration ::= LINEFLAG exp EQUALS exp SEMICOLON",
 /*  77 */ "lineflag_declaration ::= LINEFLAG exp AND exp SEMICOLON",
};
#endif /* NDEBUG */

#if YYSTACKDEPTH<=0
/*
** Try to increase the size of the parser stack.
*/
static void yyGrowStack(yyParser *p){
  int newSize;
  yyStackEntry *pNew;

  newSize = p->yystksz*2 + 100;
  pNew = realloc(p->yystack, newSize*sizeof(pNew[0]));
  if( pNew ){
    p->yystack = pNew;
    p->yystksz = newSize;
#ifndef NDEBUG
    if( yyTraceFILE ){
      fprintf(yyTraceFILE,"%sStack grows to %d entries!\n",
              yyTracePrompt, p->yystksz);
    }
#endif
  }
}
#endif

/* 
** This function allocates a new parser.
** The only argument is a pointer to a function which works like
** malloc.
**
** Inputs:
** A pointer to the function used to allocate memory.
**
** Outputs:
** A pointer to a parser.  This pointer is used in subsequent calls
** to XlatParse and XlatParseFree.
*/
void *XlatParseAlloc(void *(CDECL *mallocProc)(size_t)){
  yyParser *pParser;
  pParser = (yyParser*)(*mallocProc)( (size_t)sizeof(yyParser) );
  if( pParser ){
    pParser->yyidx = -1;
#if YYSTACKDEPTH<=0
    yyGrowStack(pParser);
#endif
  }
  return pParser;
}

/* The following function deletes the value associated with a
** symbol.  The symbol can be either a terminal or nonterminal.
** "yymajor" is the symbol code, and "yypminor" is a pointer to
** the value.
*/
static void yy_destructor(YYCODETYPE yymajor, YYMINORTYPE *yypminor){
  switch( yymajor ){
    /* Here is inserted the actions which take place when a
    ** terminal or non-terminal is destroyed.  This can happen
    ** when the symbol is popped from the stack during a
    ** reduce or during error processing or when a parser is 
    ** being destroyed before it is finished parsing.
    **
    ** Note: during a reduce, the only symbols destroyed are those
    ** which appear on the RHS of the rule, but which are not used
    ** inside the C code.
    */
    case 1: /* NOP */
    case 2: /* OR */
    case 3: /* XOR */
    case 4: /* AND */
    case 5: /* MINUS */
    case 6: /* PLUS */
    case 7: /* MULTIPLY */
    case 8: /* DIVIDE */
    case 9: /* MODULUS */
    case 10: /* NEG */
    case 11: /* NUM */
    case 12: /* LPAREN */
    case 13: /* RPAREN */
    case 14: /* DEFINE */
    case 15: /* SYM */
    case 16: /* RBRACE */
    case 17: /* ENUM */
    case 18: /* LBRACE */
    case 19: /* COMMA */
    case 20: /* EQUALS */
    case 21: /* TAG */
    case 22: /* LBRACKET */
    case 23: /* RBRACKET */
    case 24: /* FLAGS */
    case 25: /* ARG2 */
    case 26: /* ARG3 */
    case 27: /* ARG4 */
    case 28: /* ARG5 */
    case 29: /* OR_EQUAL */
    case 30: /* COLON */
    case 31: /* MAXLINESPECIAL */
    case 32: /* SEMICOLON */
    case 33: /* SECTOR */
    case 34: /* NOBITMASK */
    case 35: /* BITMASK */
    case 36: /* CLEAR */
    case 37: /* LSHASSIGN */
    case 38: /* RSHASSIGN */
    case 39: /* LINEFLAG */
#line 4 "xlat_parser.y"
{}
#line 560 "xlat_parser.c"
      break;
    default:  break;   /* If no destructor action specified: do nothing */
  }
}

/*
** Pop the parser's stack once.
**
** If there is a destructor routine associated with the token which
** is popped from the stack, then call it.
**
** Return the major token number for the symbol popped.
*/
static int yy_pop_parser_stack(yyParser *pParser){
  YYCODETYPE yymajor;
  yyStackEntry *yytos = &pParser->yystack[pParser->yyidx];

  if( pParser->yyidx<0 ) return 0;
#ifndef NDEBUG
  if( yyTraceFILE && pParser->yyidx>=0 ){
    fprintf(yyTraceFILE,"%sPopping %s\n",
      yyTracePrompt,
      yyTokenName[yytos->major]);
  }
#endif
  yymajor = yytos->major;
  yy_destructor( yymajor, &yytos->minor);
  pParser->yyidx--;
  return yymajor;
}

/* 
** Deallocate and destroy a parser.  Destructors are all called for
** all stack elements before shutting the parser down.
**
** Inputs:
** <ul>
** <li>  A pointer to the parser.  This should be a pointer
**       obtained from XlatParseAlloc.
** <li>  A pointer to a function used to reclaim memory obtained
**       from malloc.
** </ul>
*/
void XlatParseFree(
  void *p,                    /* The parser to be deleted */
  void (CDECL *freeProc)(void*)     /* Function used to reclaim memory */
){
  yyParser *pParser = (yyParser*)p;
  if( pParser==0 ) return;
  while( pParser->yyidx>=0 ) yy_pop_parser_stack(pParser);
#if YYSTACKDEPTH<=0
  free(pParser->yystack);
#endif
  (*freeProc)((void*)pParser);
}

/*
** Find the appropriate action for a parser given the terminal
** look-ahead token iLookAhead.
**
** If the look-ahead token is YYNOCODE, then check to see if the action is
** independent of the look-ahead.  If it is, return the action, otherwise
** return YY_NO_ACTION.
*/
static int yy_find_shift_action(
  yyParser *pParser,        /* The parser */
  YYCODETYPE iLookAhead     /* The look-ahead token */
){
  int i;
  int stateno = pParser->yystack[pParser->yyidx].stateno;
 
  if( stateno>YY_SHIFT_MAX || (i = yy_shift_ofst[stateno])==YY_SHIFT_USE_DFLT ){
    return yy_default[stateno];
  }
  assert( iLookAhead!=YYNOCODE );
  i += iLookAhead;
  if( i<0 || i>=YY_SZ_ACTTAB || yy_lookahead[i]!=iLookAhead ){
    if( iLookAhead>0 ){
#ifdef YYFALLBACK
      int iFallback;            /* Fallback token */
      if( iLookAhead<sizeof(yyFallback)/sizeof(yyFallback[0])
             && (iFallback = yyFallback[iLookAhead])!=0 ){
#ifndef NDEBUG
        if( yyTraceFILE ){
          fprintf(yyTraceFILE, "%sFALLBACK %s => %s\n",
             yyTracePrompt, yyTokenName[iLookAhead], yyTokenName[iFallback]);
        }
#endif
        return yy_find_shift_action(pParser, iFallback);
      }
#endif
#ifdef YYWILDCARD
      {
        int j = i - iLookAhead + YYWILDCARD;
        if( j>=0 && j<YY_SZ_ACTTAB && yy_lookahead[j]==YYWILDCARD ){
#ifndef NDEBUG
          if( yyTraceFILE ){
            fprintf(yyTraceFILE, "%sWILDCARD %s => %s\n",
               yyTracePrompt, yyTokenName[iLookAhead], yyTokenName[YYWILDCARD]);
          }
#endif /* NDEBUG */
          return yy_action[j];
        }
      }
#endif /* YYWILDCARD */
    }
    return yy_default[stateno];
  }else{
    return yy_action[i];
  }
}

/*
** Find the appropriate action for a parser given the non-terminal
** look-ahead token iLookAhead.
**
** If the look-ahead token is YYNOCODE, then check to see if the action is
** independent of the look-ahead.  If it is, return the action, otherwise
** return YY_NO_ACTION.
*/
static int yy_find_reduce_action(
  int stateno,              /* Current state number */
  YYCODETYPE iLookAhead     /* The look-ahead token */
){
  int i;
  if( stateno>YY_REDUCE_MAX ||
	  (i = yy_reduce_ofst[stateno])==YY_REDUCE_USE_DFLT ){
	return yy_default[stateno];
  }
  assert( i!=YY_REDUCE_USE_DFLT );
  assert( iLookAhead!=YYNOCODE );
  i += iLookAhead;
  if( i<0 || i>=YY_SZ_ACTTAB || yy_lookahead[i]!=iLookAhead ){
    return yy_default[stateno];
  }else{
	return yy_action[i];
  }
  return yy_action[i];
}

/*
** The following routine is called if the stack overflows.
*/
static void yyStackOverflow(yyParser *yypParser, YYMINORTYPE *yypMinor){
   XlatParseARG_FETCH;
   yypParser->yyidx--;
#ifndef NDEBUG
   if( yyTraceFILE ){
     fprintf(yyTraceFILE,"%sStack Overflow!\n",yyTracePrompt);
   }
#endif
   while( yypParser->yyidx>=0 ) yy_pop_parser_stack(yypParser);
   /* Here code is inserted which will execute if the parser
   ** stack every overflows */
   XlatParseARG_STORE; /* Suppress warning about unused %extra_argument var */
}

/*
** Perform a shift action.
*/
static void yy_shift(
  yyParser *yypParser,          /* The parser to be shifted */
  int yyNewState,               /* The new state to shift in */
  int yyMajor,                  /* The major token to shift in */
  YYMINORTYPE *yypMinor         /* Pointer ot the minor token to shift in */
){
  yyStackEntry *yytos;
  yypParser->yyidx++;
#if YYSTACKDEPTH>0
  if( yypParser->yyidx>=YYSTACKDEPTH ){
    yyStackOverflow(yypParser, yypMinor);
    return;
  }
#else
  if( yypParser->yyidx>=yypParser->yystksz ){
    yyGrowStack(yypParser);
    if( yypParser->yyidx>=yypParser->yystksz ){
      yyStackOverflow(yypParser, yypMinor);
      return;
    }
  }
#endif
  yytos = &yypParser->yystack[yypParser->yyidx];
  yytos->stateno = yyNewState;
  yytos->major = yyMajor;
  yytos->minor = *yypMinor;
#ifndef NDEBUG
  if( yyTraceFILE && yypParser->yyidx>0 ){
    int i;
    fprintf(yyTraceFILE,"%sShift %d\n",yyTracePrompt,yyNewState);
    fprintf(yyTraceFILE,"%sStack:",yyTracePrompt);
    for(i=1; i<=yypParser->yyidx; i++)
      fprintf(yyTraceFILE," (%d)%s",yypParser->yystack[i].stateno,yyTokenName[yypParser->yystack[i].major]);
    fprintf(yyTraceFILE,"\n");
  }
#endif
}

/* The following table contains information about every rule that
** is used during the reduce.
*/
static const struct {
  YYCODETYPE lhs;         /* Symbol on the left-hand side of the rule */
  unsigned char nrhs;     /* Number of right-hand side symbols in the rule */
} yyRuleInfo[] = {
  { 41, 1 },
  { 42, 0 },
  { 42, 2 },
  { 43, 1 },
  { 43, 1 },
  { 43, 1 },
  { 43, 1 },
  { 43, 1 },
  { 43, 1 },
  { 43, 1 },
  { 43, 1 },
  { 43, 1 },
  { 52, 1 },
  { 52, 3 },
  { 52, 3 },
  { 52, 3 },
  { 52, 3 },
  { 52, 3 },
  { 52, 3 },
  { 52, 3 },
  { 52, 3 },
  { 52, 2 },
  { 52, 3 },
  { 44, 5 },
  { 45, 3 },
  { 53, 2 },
  { 54, 0 },
  { 54, 1 },
  { 54, 3 },
  { 55, 1 },
  { 55, 3 },
  { 56, 1 },
  { 46, 8 },
  { 46, 8 },
  { 58, 1 },
  { 58, 1 },
  { 58, 3 },
  { 58, 3 },
  { 58, 3 },
  { 58, 3 },
  { 58, 3 },
  { 58, 3 },
  { 58, 3 },
  { 58, 3 },
  { 58, 2 },
  { 58, 3 },
  { 59, 1 },
  { 60, 1 },
  { 60, 3 },
  { 57, 0 },
  { 57, 1 },
  { 47, 11 },
  { 67, 0 },
  { 67, 2 },
  { 66, 3 },
  { 65, 1 },
  { 65, 1 },
  { 65, 1 },
  { 65, 1 },
  { 65, 1 },
  { 64, 1 },
  { 64, 1 },
  { 63, 1 },
  { 63, 4 },
  { 62, 1 },
  { 62, 3 },
  { 61, 3 },
  { 51, 4 },
  { 48, 5 },
  { 48, 5 },
  { 48, 6 },
  { 50, 6 },
  { 50, 4 },
  { 50, 5 },
  { 68, 1 },
  { 68, 1 },
  { 49, 5 },
  { 49, 5 },
};

static void yy_accept(yyParser*);  /* Forward Declaration */

/*
** Perform a reduce action and the shift that must immediately
** follow the reduce.
*/
static void yy_reduce(
  yyParser *yypParser,         /* The parser */
  int yyruleno                 /* Number of the rule by which to reduce */
){
  int yygoto;                     /* The next state */
  int yyact;                      /* The next action */
  YYMINORTYPE yygotominor;        /* The LHS of the rule reduced */
  yyStackEntry *yymsp;            /* The top of the parser's stack */
  int yysize;                     /* Amount to pop the stack */
  XlatParseARG_FETCH;
  yymsp = &yypParser->yystack[yypParser->yyidx];
#ifndef NDEBUG
  if( yyTraceFILE && yyruleno>=0 
        && yyruleno<(int)(sizeof(yyRuleName)/sizeof(yyRuleName[0])) ){
    fprintf(yyTraceFILE, "%sReduce [%s].\n", yyTracePrompt,
      yyRuleName[yyruleno]);
  }
#endif /* NDEBUG */

  /* Silence complaints from purify about yygotominor being uninitialized
  ** in some cases when it is copied into the stack after the following
  ** switch.  yygotominor is uninitialized when a rule reduces that does
  ** not set the value of its left-hand side nonterminal.  Leaving the
  ** value of the nonterminal uninitialized is utterly harmless as long
  ** as the value is never used.  So really the only thing this code
  ** accomplishes is to quieten purify.  
  **
  ** 2007-01-16:  The wireshark project (www.wireshark.org) reports that
  ** without this code, their parser segfaults.  I'm not sure what there
  ** parser is doing to make this happen.  This is the second bug report
  ** from wireshark this week.  Clearly they are stressing Lemon in ways
  ** that it has not been previously stressed...  (SQLite ticket #2172)
  */
  memset(&yygotominor, 0, sizeof(yygotominor));


  switch( yyruleno ){
  /* Beginning here are the reduction cases.  A typical example
  ** follows:
  **   case 0:
  **  #line <lineno> <grammarfile>
  **     { ... }           // User supplied code
  **  #line <lineno> <thisfile>
  **     break;
  */
      case 0: /* main ::= translation_unit */
      case 1: /*translation_unit ::= */
      case 2: /*translation_unit ::= translation_unit external_declaration */
      case 3: /*external_declaration ::= define_statement */
      case 4: /*external_declaration ::= enum_statement */
      case 5: /*external_declaration ::= linetype_declaration */
      case 6: /*external_declaration ::= boom_declaration */
      case 7: /*external_declaration ::= sector_declaration */
      case 8: /*external_declaration ::= lineflag_declaration */
      case 9: /*external_declaration ::= sector_bitmask */
      case 10: /*external_declaration ::= maxlinespecial_def */
      case 26: /*enum_list ::= */
      case 27: /*enum_list ::= single_enum */
#line 10 "xlat_parser.y"
{
}
#line 913 "xlat_parser.c"
        break;
      case 11: /* external_declaration ::= NOP */
#line 23 "xlat_parser.y"
{
  yy_destructor(1,&yymsp[0].minor);
}
#line 920 "xlat_parser.c"
        break;
      case 12: /* exp ::= NUM */
#line 34 "xlat_parser.y"
{ yygotominor.yy32 = yymsp[0].minor.yy0.val; }
#line 925 "xlat_parser.c"
        break;
      case 13: /* exp ::= exp PLUS exp */
#line 35 "xlat_parser.y"
{ yygotominor.yy32 = yymsp[-2].minor.yy32 + yymsp[0].minor.yy32;   yy_destructor(6,&yymsp[-1].minor);
}
#line 931 "xlat_parser.c"
        break;
      case 14: /* exp ::= exp MINUS exp */
#line 36 "xlat_parser.y"
{ yygotominor.yy32 = yymsp[-2].minor.yy32 - yymsp[0].minor.yy32;   yy_destructor(5,&yymsp[-1].minor);
}
#line 937 "xlat_parser.c"
        break;
      case 15: /* exp ::= exp MULTIPLY exp */
#line 37 "xlat_parser.y"
{ yygotominor.yy32 = yymsp[-2].minor.yy32 * yymsp[0].minor.yy32;   yy_destructor(7,&yymsp[-1].minor);
}
#line 943 "xlat_parser.c"
        break;
      case 16: /* exp ::= exp DIVIDE exp */
#line 38 "xlat_parser.y"
{ if (yymsp[0].minor.yy32 != 0) yygotominor.yy32 = yymsp[-2].minor.yy32 / yymsp[0].minor.yy32; else context->PrintError("Division by zero");   yy_destructor(8,&yymsp[-1].minor);
}
#line 949 "xlat_parser.c"
        break;
      case 17: /* exp ::= exp MODULUS exp */
#line 39 "xlat_parser.y"
{ if (yymsp[0].minor.yy32 != 0) yygotominor.yy32 = yymsp[-2].minor.yy32 % yymsp[0].minor.yy32; else context->PrintError("Division by zero");   yy_destructor(9,&yymsp[-1].minor);
}
#line 955 "xlat_parser.c"
        break;
      case 18: /* exp ::= exp OR exp */
#line 40 "xlat_parser.y"
{ yygotominor.yy32 = yymsp[-2].minor.yy32 | yymsp[0].minor.yy32;   yy_destructor(2,&yymsp[-1].minor);
}
#line 961 "xlat_parser.c"
        break;
      case 19: /* exp ::= exp AND exp */
#line 41 "xlat_parser.y"
{ yygotominor.yy32 = yymsp[-2].minor.yy32 & yymsp[0].minor.yy32;   yy_destructor(4,&yymsp[-1].minor);
}
#line 967 "xlat_parser.c"
        break;
      case 20: /* exp ::= exp XOR exp */
#line 42 "xlat_parser.y"
{ yygotominor.yy32 = yymsp[-2].minor.yy32 ^ yymsp[0].minor.yy32;   yy_destructor(3,&yymsp[-1].minor);
}
#line 973 "xlat_parser.c"
        break;
      case 21: /* exp ::= MINUS exp */
#line 43 "xlat_parser.y"
{ yygotominor.yy32 = -yymsp[0].minor.yy32;   yy_destructor(5,&yymsp[-1].minor);
}
#line 979 "xlat_parser.c"
        break;
      case 22: /* exp ::= LPAREN exp RPAREN */
      case 45: /*exp_with_tag ::= LPAREN exp_with_tag RPAREN */
#line 44 "xlat_parser.y"
{ yygotominor.yy32 = yymsp[-1].minor.yy32;   yy_destructor(12,&yymsp[-2].minor);
  yy_destructor(13,&yymsp[0].minor);
}
#line 987 "xlat_parser.c"
        break;
      case 23: /* define_statement ::= DEFINE SYM LPAREN exp RPAREN */
#line 54 "xlat_parser.y"
{
	context->AddSym (yymsp[-3].minor.yy0.sym, yymsp[-1].minor.yy32);
  yy_destructor(14,&yymsp[-4].minor);
  yy_destructor(12,&yymsp[-2].minor);
  yy_destructor(13,&yymsp[0].minor);
}
#line 997 "xlat_parser.c"
        break;
      case 24: /* enum_statement ::= enum_open enum_list RBRACE */
#line 64 "xlat_parser.y"
{
  yy_destructor(16,&yymsp[0].minor);
}
#line 1004 "xlat_parser.c"
        break;
      case 25: /* enum_open ::= ENUM LBRACE */
#line 67 "xlat_parser.y"
{
	context->EnumVal = 0;
  yy_destructor(17,&yymsp[-1].minor);
  yy_destructor(18,&yymsp[0].minor);
}
#line 1013 "xlat_parser.c"
        break;
      case 28: /* enum_list ::= enum_list COMMA single_enum */
#line 73 "xlat_parser.y"
{
  yy_destructor(19,&yymsp[-1].minor);
}
#line 1020 "xlat_parser.c"
        break;
      case 29: /* single_enum ::= SYM */
#line 76 "xlat_parser.y"
{
	context->AddSym (yymsp[0].minor.yy0.sym, context->EnumVal++);
}
#line 1027 "xlat_parser.c"
        break;
      case 30: /* single_enum ::= SYM EQUALS exp */
#line 81 "xlat_parser.y"
{
	context->AddSym (yymsp[-2].minor.yy0.sym, yymsp[0].minor.yy32);
	context->EnumVal = yymsp[0].minor.yy32+1;
  yy_destructor(20,&yymsp[-1].minor);
}
#line 1036 "xlat_parser.c"
        break;
      case 31: /* linetype_exp ::= exp */
#line 94 "xlat_parser.y"
{
	yygotominor.yy32 = static_cast<XlatParseContext *>(context)->DefiningLineType = yymsp[0].minor.yy32;
}
#line 1043 "xlat_parser.c"
        break;
      case 32: /* linetype_declaration ::= linetype_exp EQUALS exp COMMA exp LPAREN special_args RPAREN */
#line 99 "xlat_parser.y"
{
	SimpleLineTranslations.SetVal(yymsp[-7].minor.yy32, 
		FLineTrans(yymsp[-3].minor.yy32&0xffff, yymsp[-5].minor.yy32+yymsp[-1].minor.yy7.addflags, yymsp[-1].minor.yy7.args[0], yymsp[-1].minor.yy7.args[1], yymsp[-1].minor.yy7.args[2], yymsp[-1].minor.yy7.args[3], yymsp[-1].minor.yy7.args[4]));
	static_cast<XlatParseContext *>(context)->DefiningLineType = -1;
  yy_destructor(20,&yymsp[-6].minor);
  yy_destructor(19,&yymsp[-4].minor);
  yy_destructor(12,&yymsp[-2].minor);
  yy_destructor(13,&yymsp[0].minor);
}
#line 1056 "xlat_parser.c"
        break;
      case 33: /* linetype_declaration ::= linetype_exp EQUALS exp COMMA SYM LPAREN special_args RPAREN */
#line 106 "xlat_parser.y"
{
	Printf ("%s, line %d: %s is undefined\n", context->SourceFile, context->SourceLine, yymsp[-3].minor.yy0.sym);
	static_cast<XlatParseContext *>(context)->DefiningLineType = -1;
  yy_destructor(20,&yymsp[-6].minor);
  yy_destructor(19,&yymsp[-4].minor);
  yy_destructor(12,&yymsp[-2].minor);
  yy_destructor(13,&yymsp[0].minor);
}
#line 1068 "xlat_parser.c"
        break;
      case 34: /* exp_with_tag ::= NUM */
#line 112 "xlat_parser.y"
{ XlatExpressions.Push(yymsp[0].minor.yy0.val); yygotominor.yy32 = XlatExpressions.Push(XEXP_Const); }
#line 1073 "xlat_parser.c"
        break;
      case 35: /* exp_with_tag ::= TAG */
#line 113 "xlat_parser.y"
{ yygotominor.yy32 = XlatExpressions.Push(XEXP_Tag);   yy_destructor(21,&yymsp[0].minor);
}
#line 1079 "xlat_parser.c"
        break;
      case 36: /* exp_with_tag ::= exp_with_tag PLUS exp_with_tag */
#line 114 "xlat_parser.y"
{ yygotominor.yy32 = XlatExpressions.Push(XEXP_Add);   yy_destructor(6,&yymsp[-1].minor);
}
#line 1085 "xlat_parser.c"
        break;
      case 37: /* exp_with_tag ::= exp_with_tag MINUS exp_with_tag */
#line 115 "xlat_parser.y"
{ yygotominor.yy32 = XlatExpressions.Push(XEXP_Sub);   yy_destructor(5,&yymsp[-1].minor);
}
#line 1091 "xlat_parser.c"
        break;
      case 38: /* exp_with_tag ::= exp_with_tag MULTIPLY exp_with_tag */
#line 116 "xlat_parser.y"
{ yygotominor.yy32 = XlatExpressions.Push(XEXP_Mul);   yy_destructor(7,&yymsp[-1].minor);
}
#line 1097 "xlat_parser.c"
        break;
      case 39: /* exp_with_tag ::= exp_with_tag DIVIDE exp_with_tag */
#line 117 "xlat_parser.y"
{ yygotominor.yy32 = XlatExpressions.Push(XEXP_Div);   yy_destructor(8,&yymsp[-1].minor);
}
#line 1103 "xlat_parser.c"
        break;
      case 40: /* exp_with_tag ::= exp_with_tag MODULUS exp_with_tag */
#line 118 "xlat_parser.y"
{ yygotominor.yy32 = XlatExpressions.Push(XEXP_Mod);   yy_destructor(9,&yymsp[-1].minor);
}
#line 1109 "xlat_parser.c"
        break;
      case 41: /* exp_with_tag ::= exp_with_tag OR exp_with_tag */
#line 119 "xlat_parser.y"
{ yygotominor.yy32 = XlatExpressions.Push(XEXP_Or);    yy_destructor(2,&yymsp[-1].minor);
}
#line 1115 "xlat_parser.c"
        break;
      case 42: /* exp_with_tag ::= exp_with_tag AND exp_with_tag */
#line 120 "xlat_parser.y"
{ yygotominor.yy32 = XlatExpressions.Push(XEXP_And);   yy_destructor(4,&yymsp[-1].minor);
}
#line 1121 "xlat_parser.c"
        break;
      case 43: /* exp_with_tag ::= exp_with_tag XOR exp_with_tag */
#line 121 "xlat_parser.y"
{ yygotominor.yy32 = XlatExpressions.Push(XEXP_Xor);   yy_destructor(3,&yymsp[-1].minor);
}
#line 1127 "xlat_parser.c"
        break;
      case 44: /* exp_with_tag ::= MINUS exp_with_tag */
#line 122 "xlat_parser.y"
{ yygotominor.yy32 = XlatExpressions.Push(XEXP_Neg);   yy_destructor(5,&yymsp[-1].minor);
}
#line 1133 "xlat_parser.c"
        break;
      case 46: /* special_arg ::= exp_with_tag */
#line 129 "xlat_parser.y"
{
	if (XlatExpressions[yymsp[0].minor.yy32] == XEXP_Tag)
	{ // Store tags directly
		yygotominor.yy120.arg = 0;
		yygotominor.yy120.argop = ARGOP_Tag;
		XlatExpressions.Delete(yymsp[0].minor.yy32);
	}
	else
	{ // Try and evaluate it. If it's a constant, store it and erase the
	  // expression. Otherwise, store the index to the expression. We make
	  // no attempt to simplify non-constant expressions.
		FXlatExprState state;
		int val;
		const int *endpt;
		int *xnode;
		
		state.linetype = static_cast<XlatParseContext *>(context)->DefiningLineType;
		state.tag = 0;
		state.bIsConstant = true;
		xnode = &XlatExpressions[yymsp[0].minor.yy32];
		endpt = XlatExprEval[*xnode](&val, xnode, &state);
		if (state.bIsConstant)
		{
			yygotominor.yy120.arg = val;
			yygotominor.yy120.argop = ARGOP_Const;
			endpt++;
			assert(endpt >= &XlatExpressions[0]);
			XlatExpressions.Resize((unsigned)(endpt - &XlatExpressions[0]));
		}
		else
		{
			yygotominor.yy120.arg = yymsp[0].minor.yy32;
			yygotominor.yy120.argop = ARGOP_Expr;
		}
	}
}
#line 1173 "xlat_parser.c"
        break;
      case 47: /* multi_special_arg ::= special_arg */
#line 169 "xlat_parser.y"
{
	yygotominor.yy7.addflags = yymsp[0].minor.yy120.argop << LINETRANS_TAGSHIFT;
	yygotominor.yy7.argcount = 1;
	yygotominor.yy7.args[0] = yymsp[0].minor.yy120.arg;
	yygotominor.yy7.args[1] = 0;
	yygotominor.yy7.args[2] = 0;
	yygotominor.yy7.args[3] = 0;
	yygotominor.yy7.args[4] = 0;
}
#line 1186 "xlat_parser.c"
        break;
      case 48: /* multi_special_arg ::= multi_special_arg COMMA special_arg */
#line 179 "xlat_parser.y"
{
	yygotominor.yy7 = yymsp[-2].minor.yy7;
	if (yygotominor.yy7.argcount < LINETRANS_MAXARGS)
	{
		yygotominor.yy7.addflags |= yymsp[0].minor.yy120.argop << (LINETRANS_TAGSHIFT + yygotominor.yy7.argcount * TAGOP_NUMBITS);
		yygotominor.yy7.args[yygotominor.yy7.argcount] = yymsp[0].minor.yy120.arg;
		yygotominor.yy7.argcount++;
	}
	else if (yygotominor.yy7.argcount++ == LINETRANS_MAXARGS)
	{
		context->PrintError("Line special has too many arguments\n");
	}
  yy_destructor(19,&yymsp[-1].minor);
}
#line 1204 "xlat_parser.c"
        break;
      case 49: /* special_args ::= */
#line 196 "xlat_parser.y"
{
	yygotominor.yy7.addflags = 0;
	yygotominor.yy7.argcount = 0;
	yygotominor.yy7.args[0] = 0;
	yygotominor.yy7.args[1] = 0;
	yygotominor.yy7.args[2] = 0;
	yygotominor.yy7.args[3] = 0;
	yygotominor.yy7.args[4] = 0;
}
#line 1217 "xlat_parser.c"
        break;
      case 50: /* special_args ::= multi_special_arg */
#line 206 "xlat_parser.y"
{
	yygotominor.yy7 = yymsp[0].minor.yy7;
}
#line 1224 "xlat_parser.c"
        break;
      case 51: /* boom_declaration ::= LBRACKET exp RBRACKET LPAREN exp COMMA exp RPAREN LBRACE boom_body RBRACE */
#line 226 "xlat_parser.y"
{
	int i;
	MoreLines *probe;

	if (NumBoomish == MAX_BOOMISH)
	{
		MoreLines *probe = yymsp[-1].minor.yy129;

		while (probe != NULL)
		{
			MoreLines *next = probe->next;
			delete probe;
			probe = next;
		}
		Printf ("%s, line %d: Too many BOOM translators\n", context->SourceFile, context->SourceLine);
	}
	else
	{
		Boomish[NumBoomish].FirstLinetype = yymsp[-6].minor.yy32;
		Boomish[NumBoomish].LastLinetype = yymsp[-4].minor.yy32;
		Boomish[NumBoomish].NewSpecial = yymsp[-9].minor.yy32;
		
		for (i = 0, probe = yymsp[-1].minor.yy129; probe != NULL; i++)
		{
			MoreLines *next = probe->next;
			Boomish[NumBoomish].Args.Push(probe->arg);
			delete probe;
			probe = next;
		}
		NumBoomish++;
	}
  yy_destructor(22,&yymsp[-10].minor);
  yy_destructor(23,&yymsp[-8].minor);
  yy_destructor(12,&yymsp[-7].minor);
  yy_destructor(19,&yymsp[-5].minor);
  yy_destructor(13,&yymsp[-3].minor);
  yy_destructor(18,&yymsp[-2].minor);
  yy_destructor(16,&yymsp[0].minor);
}
#line 1267 "xlat_parser.c"
        break;
      case 52: /* boom_body ::= */
#line 260 "xlat_parser.y"
{
	yygotominor.yy129 = NULL;
}
#line 1274 "xlat_parser.c"
        break;
      case 53: /* boom_body ::= boom_line boom_body */
#line 264 "xlat_parser.y"
{
	yygotominor.yy129 = new MoreLines;
	yygotominor.yy129->next = yymsp[0].minor.yy129;
	yygotominor.yy129->arg = yymsp[-1].minor.yy130;
}
#line 1283 "xlat_parser.c"
        break;
      case 54: /* boom_line ::= boom_selector boom_op boom_args */
#line 271 "xlat_parser.y"
{
	yygotominor.yy130.bOrExisting = (yymsp[-1].minor.yy32 == OR_EQUAL);
	yygotominor.yy130.bUseConstant = (yymsp[0].minor.yy63.filters == NULL);
	yygotominor.yy130.ArgNum = yymsp[-2].minor.yy32;
	yygotominor.yy130.ConstantValue = yymsp[0].minor.yy63.constant;
	yygotominor.yy130.AndValue = yymsp[0].minor.yy63.mask;

	if (yymsp[0].minor.yy63.filters != NULL)
	{
		int i;
		MoreFilters *probe;

		for (i = 0, probe = yymsp[0].minor.yy63.filters; probe != NULL; i++)
		{
			MoreFilters *next = probe->next;
			if (i < 15)
			{
				yygotominor.yy130.ResultFilter[i] = probe->filter.filter;
				yygotominor.yy130.ResultValue[i] = probe->filter.value;
			}
			else if (i == 15)
			{
				context->PrintError ("Lists can only have 15 elements");
			}
			delete probe;
			probe = next;
		}
		yygotominor.yy130.ListSize = i > 15 ? 15 : i;
	}
}
#line 1317 "xlat_parser.c"
        break;
      case 55: /* boom_selector ::= FLAGS */
#line 302 "xlat_parser.y"
{ yygotominor.yy32 = 4;   yy_destructor(24,&yymsp[0].minor);
}
#line 1323 "xlat_parser.c"
        break;
      case 56: /* boom_selector ::= ARG2 */
#line 303 "xlat_parser.y"
{ yygotominor.yy32 = 0;   yy_destructor(25,&yymsp[0].minor);
}
#line 1329 "xlat_parser.c"
        break;
      case 57: /* boom_selector ::= ARG3 */
#line 304 "xlat_parser.y"
{ yygotominor.yy32 = 1;   yy_destructor(26,&yymsp[0].minor);
}
#line 1335 "xlat_parser.c"
        break;
      case 58: /* boom_selector ::= ARG4 */
#line 305 "xlat_parser.y"
{ yygotominor.yy32 = 2;   yy_destructor(27,&yymsp[0].minor);
}
#line 1341 "xlat_parser.c"
        break;
      case 59: /* boom_selector ::= ARG5 */
#line 306 "xlat_parser.y"
{ yygotominor.yy32 = 3;   yy_destructor(28,&yymsp[0].minor);
}
#line 1347 "xlat_parser.c"
        break;
      case 60: /* boom_op ::= EQUALS */
#line 308 "xlat_parser.y"
{ yygotominor.yy32 = '=';   yy_destructor(20,&yymsp[0].minor);
}
#line 1353 "xlat_parser.c"
        break;
      case 61: /* boom_op ::= OR_EQUAL */
#line 309 "xlat_parser.y"
{ yygotominor.yy32 = OR_EQUAL;   yy_destructor(29,&yymsp[0].minor);
}
#line 1359 "xlat_parser.c"
        break;
      case 62: /* boom_args ::= exp */
#line 312 "xlat_parser.y"
{
	yygotominor.yy63.constant = yymsp[0].minor.yy32;
	yygotominor.yy63.filters = NULL;
}
#line 1367 "xlat_parser.c"
        break;
      case 63: /* boom_args ::= exp LBRACKET arg_list RBRACKET */
#line 317 "xlat_parser.y"
{
	yygotominor.yy63.mask = yymsp[-3].minor.yy32;
	yygotominor.yy63.filters = yymsp[-1].minor.yy8;
  yy_destructor(22,&yymsp[-2].minor);
  yy_destructor(23,&yymsp[0].minor);
}
#line 1377 "xlat_parser.c"
        break;
      case 64: /* arg_list ::= list_val */
#line 323 "xlat_parser.y"
{
	yygotominor.yy8 = new MoreFilters;
	yygotominor.yy8->next = NULL;
	yygotominor.yy8->filter = yymsp[0].minor.yy83;
}
#line 1386 "xlat_parser.c"
        break;
      case 65: /* arg_list ::= list_val COMMA arg_list */
#line 329 "xlat_parser.y"
{
	yygotominor.yy8 = new MoreFilters;
	yygotominor.yy8->next = yymsp[0].minor.yy8;
	yygotominor.yy8->filter = yymsp[-2].minor.yy83;
  yy_destructor(19,&yymsp[-1].minor);
}
#line 1396 "xlat_parser.c"
        break;
      case 66: /* list_val ::= exp COLON exp */
#line 336 "xlat_parser.y"
{
	yygotominor.yy83.filter = yymsp[-2].minor.yy32;
	yygotominor.yy83.value = yymsp[0].minor.yy32;
  yy_destructor(30,&yymsp[-1].minor);
}
#line 1405 "xlat_parser.c"
        break;
      case 67: /* maxlinespecial_def ::= MAXLINESPECIAL EQUALS exp SEMICOLON */
#line 348 "xlat_parser.y"
{
	// Just kill all specials higher than the max.
	// If the translator wants to redefine some later, just let it.
	SimpleLineTranslations.Resize(yymsp[-1].minor.yy32+1);
  yy_destructor(31,&yymsp[-3].minor);
  yy_destructor(20,&yymsp[-2].minor);
  yy_destructor(32,&yymsp[0].minor);
}
#line 1417 "xlat_parser.c"
        break;
      case 68: /* sector_declaration ::= SECTOR exp EQUALS exp SEMICOLON */
#line 363 "xlat_parser.y"
{
	FSectorTrans tr(yymsp[-1].minor.yy32, true);
	SectorTranslations.SetVal(yymsp[-3].minor.yy32, tr);
  yy_destructor(33,&yymsp[-4].minor);
  yy_destructor(20,&yymsp[-2].minor);
  yy_destructor(32,&yymsp[0].minor);
}
#line 1428 "xlat_parser.c"
        break;
      case 69: /* sector_declaration ::= SECTOR exp EQUALS SYM SEMICOLON */
#line 369 "xlat_parser.y"
{
	Printf("Unknown constant '%s'\n", yymsp[-1].minor.yy0.sym);
  yy_destructor(33,&yymsp[-4].minor);
  yy_destructor(20,&yymsp[-2].minor);
  yy_destructor(32,&yymsp[0].minor);
}
#line 1438 "xlat_parser.c"
        break;
      case 70: /* sector_declaration ::= SECTOR exp EQUALS exp NOBITMASK SEMICOLON */
#line 374 "xlat_parser.y"
{
	FSectorTrans tr(yymsp[-2].minor.yy32, false);
	SectorTranslations.SetVal(yymsp[-4].minor.yy32, tr);
  yy_destructor(33,&yymsp[-5].minor);
  yy_destructor(20,&yymsp[-3].minor);
  yy_destructor(34,&yymsp[-1].minor);
  yy_destructor(32,&yymsp[0].minor);
}
#line 1450 "xlat_parser.c"
        break;
      case 71: /* sector_bitmask ::= SECTOR BITMASK exp sector_op exp SEMICOLON */
#line 380 "xlat_parser.y"
{
	FSectorMask sm = { yymsp[-3].minor.yy32, yymsp[-2].minor.yy32, yymsp[-1].minor.yy32};
	SectorMasks.Push(sm);
  yy_destructor(33,&yymsp[-5].minor);
  yy_destructor(35,&yymsp[-4].minor);
  yy_destructor(32,&yymsp[0].minor);
}
#line 1461 "xlat_parser.c"
        break;
      case 72: /* sector_bitmask ::= SECTOR BITMASK exp SEMICOLON */
#line 386 "xlat_parser.y"
{
	FSectorMask sm = { yymsp[-1].minor.yy32, 0, 0};
	SectorMasks.Push(sm);
  yy_destructor(33,&yymsp[-3].minor);
  yy_destructor(35,&yymsp[-2].minor);
  yy_destructor(32,&yymsp[0].minor);
}
#line 1472 "xlat_parser.c"
        break;
      case 73: /* sector_bitmask ::= SECTOR BITMASK exp CLEAR SEMICOLON */
#line 392 "xlat_parser.y"
{
	FSectorMask sm = { yymsp[-2].minor.yy32, 0, 1};
	SectorMasks.Push(sm);
  yy_destructor(33,&yymsp[-4].minor);
  yy_destructor(35,&yymsp[-3].minor);
  yy_destructor(36,&yymsp[-1].minor);
  yy_destructor(32,&yymsp[0].minor);
}
#line 1484 "xlat_parser.c"
        break;
      case 74: /* sector_op ::= LSHASSIGN */
#line 397 "xlat_parser.y"
{ yygotominor.yy32 = 1;   yy_destructor(37,&yymsp[0].minor);
}
#line 1490 "xlat_parser.c"
        break;
      case 75: /* sector_op ::= RSHASSIGN */
#line 398 "xlat_parser.y"
{ yygotominor.yy32 = -1;   yy_destructor(38,&yymsp[0].minor);
}
#line 1496 "xlat_parser.c"
        break;
      case 76: /* lineflag_declaration ::= LINEFLAG exp EQUALS exp SEMICOLON */
#line 403 "xlat_parser.y"
{
	if (yymsp[-3].minor.yy32 >= 0 && yymsp[-3].minor.yy32 < 16)
	{
		LineFlagTranslations[yymsp[-3].minor.yy32].newvalue = yymsp[-1].minor.yy32;
		LineFlagTranslations[yymsp[-3].minor.yy32].ismask = false;
	}
  yy_destructor(39,&yymsp[-4].minor);
  yy_destructor(20,&yymsp[-2].minor);
  yy_destructor(32,&yymsp[0].minor);
}
#line 1510 "xlat_parser.c"
        break;
      case 77: /* lineflag_declaration ::= LINEFLAG exp AND exp SEMICOLON */
#line 412 "xlat_parser.y"
{
	if (yymsp[-3].minor.yy32 >= 0 && yymsp[-3].minor.yy32 < 16)
	{
		LineFlagTranslations[yymsp[-3].minor.yy32].newvalue = yymsp[-1].minor.yy32;
		LineFlagTranslations[yymsp[-3].minor.yy32].ismask = true;
	}
  yy_destructor(39,&yymsp[-4].minor);
  yy_destructor(4,&yymsp[-2].minor);
  yy_destructor(32,&yymsp[0].minor);
}
#line 1524 "xlat_parser.c"
        break;
  };
  yygoto = yyRuleInfo[yyruleno].lhs;
  yysize = yyRuleInfo[yyruleno].nrhs;
  yypParser->yyidx -= yysize;
  yyact = yy_find_reduce_action(yymsp[-yysize].stateno,yygoto);
  if( yyact < YYNSTATE ){
#ifdef NDEBUG
    /* If we are not debugging and the reduce action popped at least
    ** one element off the stack, then we can push the new element back
    ** onto the stack here, and skip the stack overflow test in yy_shift().
    ** That gives a significant speed improvement. */
    if( yysize ){
      yypParser->yyidx++;
      yymsp -= yysize-1;
      yymsp->stateno = yyact;
      yymsp->major = yygoto;
      yymsp->minor = yygotominor;
    }else
#endif
    {
      yy_shift(yypParser,yyact,yygoto,&yygotominor);
    }
  }else{
    assert( yyact == YYNSTATE + YYNRULE + 1 );
    yy_accept(yypParser);
  }
}

/*
** The following code executes when the parse fails
*/
static void yy_parse_failed(
  yyParser *yypParser           /* The parser */
){
  XlatParseARG_FETCH;
#ifndef NDEBUG
  if( yyTraceFILE ){
    fprintf(yyTraceFILE,"%sFail!\n",yyTracePrompt);
  }
#endif
  while( yypParser->yyidx>=0 ) yy_pop_parser_stack(yypParser);
  /* Here code is inserted which will be executed whenever the
  ** parser fails */
  XlatParseARG_STORE; /* Suppress warning about unused %extra_argument variable */
}

/*
** The following code executes when a syntax error first occurs.
*/
static void yy_syntax_error(
  yyParser *yypParser,           /* The parser */
  int yymajor,                   /* The major type of the error token */
  YYMINORTYPE yyminor            /* The minor type of the error token */
){
  XlatParseARG_FETCH;
#define TOKEN (yyminor.yy0)
#line 7 "xlat_parser.y"
 context->PrintError("syntax error");
#line 1584 "xlat_parser.c"
  XlatParseARG_STORE; /* Suppress warning about unused %extra_argument variable */
}

/*
** The following is executed when the parser accepts
*/
static void yy_accept(
  yyParser *yypParser           /* The parser */
){
  XlatParseARG_FETCH;
#ifndef NDEBUG
  if( yyTraceFILE ){
    fprintf(yyTraceFILE,"%sAccept!\n",yyTracePrompt);
  }
#endif
  while( yypParser->yyidx>=0 ) yy_pop_parser_stack(yypParser);
  /* Here code is inserted which will be executed whenever the
  ** parser accepts */
  XlatParseARG_STORE; /* Suppress warning about unused %extra_argument variable */
}

/* The main parser program.
** The first argument is a pointer to a structure obtained from
** "XlatParseAlloc" which describes the current state of the parser.
** The second argument is the major token number.  The third is
** the minor token.  The fourth optional argument is whatever the
** user wants (and specified in the grammar) and is available for
** use by the action routines.
**
** Inputs:
** <ul>
** <li> A pointer to the parser (an opaque structure.)
** <li> The major token number.
** <li> The minor token number.
** <li> An option argument of a grammar-specified type.
** </ul>
**
** Outputs:
** None.
*/
void XlatParse(
  void *yyp,                   /* The parser */
  int yymajor,                 /* The major token code number */
  XlatParseTOKENTYPE yyminor       /* The value for the token */
  XlatParseARG_PDECL               /* Optional %extra_argument parameter */
){
  YYMINORTYPE yyminorunion;
  int yyact;            /* The parser action. */
  int yyendofinput;     /* True if we are at the end of input */
#ifdef YYERRORSYMBOL
  int yyerrorhit = 0;   /* True if yymajor has invoked an error */
#endif
  yyParser *yypParser;  /* The parser */

  /* (re)initialize the parser, if necessary */
  yypParser = (yyParser*)yyp;
  if( yypParser->yyidx<0 ){
#if YYSTACKDEPTH<=0
    if( yypParser->yystksz <=0 ){
      memset(&yyminorunion, 0, sizeof(yyminorunion));
      yyStackOverflow(yypParser, &yyminorunion);
      return;
    }
#endif
    yypParser->yyidx = 0;
    yypParser->yyerrcnt = -1;
    yypParser->yystack[0].stateno = 0;
    yypParser->yystack[0].major = 0;
  }
  yyminorunion.yy0 = yyminor;
  yyendofinput = (yymajor==0);
  XlatParseARG_STORE;

#ifndef NDEBUG
  if( yyTraceFILE ){
    fprintf(yyTraceFILE,"%sInput %s\n",yyTracePrompt,yyTokenName[yymajor]);
  }
#endif

  do{
    yyact = yy_find_shift_action(yypParser,yymajor);
    if( yyact<YYNSTATE ){
      assert( !yyendofinput );  /* Impossible to shift the $ token */
      yy_shift(yypParser,yyact,yymajor,&yyminorunion);
      yypParser->yyerrcnt--;
      yymajor = YYNOCODE;
    }else if( yyact < YYNSTATE + YYNRULE ){
      yy_reduce(yypParser,yyact-YYNSTATE);
    }else{
#ifdef YYERRORSYMBOL
      int yymx;
#endif
      assert( yyact == YY_ERROR_ACTION );
#ifndef NDEBUG
      if( yyTraceFILE ){
        fprintf(yyTraceFILE,"%sSyntax Error!\n",yyTracePrompt);
      }
#endif
#ifdef YYERRORSYMBOL
      /* A syntax error has occurred.
      ** The response to an error depends upon whether or not the
      ** grammar defines an error token "ERROR".  
      **
      ** This is what we do if the grammar does define ERROR:
      **
      **  * Call the %syntax_error function.
      **
      **  * Begin popping the stack until we enter a state where
      **    it is legal to shift the error symbol, then shift
      **    the error symbol.
      **
      **  * Set the error count to three.
      **
      **  * Begin accepting and shifting new tokens.  No new error
      **    processing will occur until three tokens have been
      **    shifted successfully.
      **
      */
      if( yypParser->yyerrcnt<0 ){
        yy_syntax_error(yypParser,yymajor,yyminorunion);
      }
      yymx = yypParser->yystack[yypParser->yyidx].major;
      if( yymx==YYERRORSYMBOL || yyerrorhit ){
#ifndef NDEBUG
        if( yyTraceFILE ){
          fprintf(yyTraceFILE,"%sDiscard input token %s\n",
             yyTracePrompt,yyTokenName[yymajor]);
        }
#endif
        yy_destructor(yymajor,&yyminorunion);
        yymajor = YYNOCODE;
      }else{
         while(
          yypParser->yyidx >= 0 &&
          yymx != YYERRORSYMBOL &&
          (yyact = yy_find_reduce_action(
                        yypParser->yystack[yypParser->yyidx].stateno,
                        YYERRORSYMBOL)) >= YYNSTATE
        ){
          yy_pop_parser_stack(yypParser);
        }
        if( yypParser->yyidx < 0 || yymajor==0 ){
          yy_destructor(yymajor,&yyminorunion);
          yy_parse_failed(yypParser);
          yymajor = YYNOCODE;
        }else if( yymx!=YYERRORSYMBOL ){
          YYMINORTYPE u2;
          u2.YYERRSYMDT = 0;
          yy_shift(yypParser,yyact,YYERRORSYMBOL,&u2);
        }
      }
      yypParser->yyerrcnt = 3;
      yyerrorhit = 1;
#else  /* YYERRORSYMBOL is not defined */
      /* This is what we do if the grammar does not define ERROR:
      **
      **  * Report an error message, and throw away the input token.
      **
      **  * If the input token is $, then fail the parse.
      **
      ** As before, subsequent error messages are suppressed until
      ** three input tokens have been successfully shifted.
      */
      if( yypParser->yyerrcnt<=0 ){
        yy_syntax_error(yypParser,yymajor,yyminorunion);
      }
      yypParser->yyerrcnt = 3;
      yy_destructor(yymajor,&yyminorunion);
      if( yyendofinput ){
        yy_parse_failed(yypParser);
      }
      yymajor = YYNOCODE;
#endif
    }
  }while( yymajor!=YYNOCODE && yypParser->yyidx>=0 );
  return;
}
