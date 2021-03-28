import com.github.javaparser.JavaParser;
import com.github.javaparser.ParseResult;
import com.github.javaparser.ast.CompilationUnit;
import com.github.javaparser.ast.Node;
import com.google.gson.Gson;
import com.google.gson.GsonBuilder;

import java.util.Scanner;

public class ASTParser {
    /**
     * Simple wrapper that outputs JSON ASTs for Java methods.
     * java-parser can only take complete classes. Hence, this wrapper creates a "fake" class around the provided
     * input Java method.
     *
     * Build file with
     * mvn clean compile assembly:single
     *
     * Use by piping a complete Java method into the .jar, e.g.,
     * >> echo public void test() { call(param); } | java -jar java-parser.jar
     *
     * Example output:
     * {
     *  "type":"MethodDeclaration",
     *  "sourceRange":{
     *      "begin":{
     *          "line":1,
     *          "column":1
     *      },
     *      "end":{
     *          "line":1,
     *          "column":36
     *      }
     *  },
     *  "childNodes":[
     *      {
     *          "type":"Modifier",
     *          "sourceRange":{
     *              "begin":{
     *                  "line":1,
     *                  "column":1
     *              },
     *              "end":{
     *                  "line":1,
     *                  "column":7
     *              }
     *          },
     *          "childNodes":[]
     *      },
     *      {
     *          "type":"SimpleName",
     *          "sourceRange":{
     *              "begin":{
     *                  "line":1,
     *                  "column":13
     *              },
     *              "end":{
     *                  "line":1,
     *                  "column":17
     *              }
     *          },
     *          "childNodes":[]
     *      },
     *      {
     *          "type":"VoidType",
     *          "sourceRange":{
     *              "begin":{
     *                  "line":1,
     *                  "column":8
     *              },
     *              "end":{
     *                  "line":1,
     *                  "column":12
     *              }
     *          },
     *          "childNodes":[]
     *      },
     *      {
     *          "type":"BlockStmt",
     *              "sourceRange":{
     *                  "begin":{
     *                      "line":1,
     *                      "column":20
     *                  },
     *              "end":{
     *                  "line":1,
     *                  "column":36
     *              }
     *          },
     *          "childNodes":[
     *              {
     *                  "type":"ExpressionStmt",
     *                  "sourceRange":{
     *                      "begin":{
     *                          "line":1,
     *                          "column":22
     *                      },
     *                      "end":{
     *                          "line":1,
     *                          "column":34
     *                      }
     *                  },
     *                  "childNodes":[
     *                      {
     *                          "type":"MethodCallExpr",
     *                          "sourceRange":{
     *                              "begin":{
     *                                  "line":1,
     *                                  "column":22
     *                              },
     *                              "end":{
     *                                  "line":1,
     *                                  "column":33
     *                              }
     *                          },
     *                          "childNodes":[
     *                              {
     *                                  "type":"SimpleName",
     *                                  "sourceRange":{
     *                                      "begin":{
     *                                          "line":1,
     *                                          "column":22
     *                                      },
     *                                      "end":{
     *                                          "line":1,
     *                                          "column":26
     *                                      }
     *                                  },
     *                                  "childNodes":[]
     *                              },
     *                              {
     *                                  "type":"NameExpr",
     *                                  "sourceRange":{
     *                                      "begin":{
     *                                          "line":1,
     *                                          "column":27
     *                                      },
     *                                      "end":{
     *                                          "line":1,
     *                                          "column":32
     *                                      }
     *                                  },
     *                                  "childNodes":[
     *                                      {
     *                                          "type":"SimpleName",
     *                                          "sourceRange":{
     *                                              "begin":{
     *                                                  "line":1,
     *                                                  "column":27
     *                                              },
     *                                              "end":{
     *                                                  "line":1,
     *                                                  "column":32
     *                                              }
     *                                          },
     *                                          "childNodes":[]
     *                                      }
     *                                  ]
     *                              }
     *                          ]
     *                      }
     *                  ]
     *              }
     *          ]
     *      }
     *  ]
     * }
     */

    public static void main(String[] args) throws Exception {
        StringBuilder codeSnippetBuilder = new StringBuilder();
        codeSnippetBuilder.append("public class Test {\n");
        Scanner sc = new Scanner(System.in);
        while (sc.hasNextLine()) {
            codeSnippetBuilder.append(sc.nextLine());
            codeSnippetBuilder.append('\n');
        }

        codeSnippetBuilder.append('}');

        JavaParser parser = new JavaParser();
        ParseResult<CompilationUnit> result = parser.parse(codeSnippetBuilder.toString());

        if (!result.getProblems().isEmpty()) {
            throw new Exception(result.getProblem(0).getVerboseMessage());
        }
        Node rootNode = result.getResult().get().getChildNodes().get(0).getChildNodes().get(2);

        Gson gsonBuilder = new GsonBuilder().create();
        String json = gsonBuilder.toJson(ASTNode.fromNode(rootNode));
        System.out.println(json);
    }

}
