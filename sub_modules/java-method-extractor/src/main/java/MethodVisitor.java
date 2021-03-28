import com.github.javaparser.ast.body.MethodDeclaration;
import com.github.javaparser.ast.visitor.VoidVisitorAdapter;

import java.util.ArrayList;
import java.util.Arrays;

public class MethodVisitor extends VoidVisitorAdapter<Object> {

    private final ArrayList<MethodContent> methods = new ArrayList<>();
    private final CommandLineValues commandLineValues;

    public MethodVisitor(CommandLineValues commandLineValues) {
        this.commandLineValues = commandLineValues;
    }

    @Override
    public void visit(MethodDeclaration node, Object arg) {
        String methodCode = node.toString();
        String methodName = node.getName();
        String doc = null;
        if (node.getJavaDoc() != null) {
            doc = node.getJavaDoc().getContent();
        } else if (node.getComment() != null) {
            doc = node.getComment().getContent();
        } else if (node.getParentNode().getComment() != null) {
            doc = node.getParentNode().getComment().getContent();
        }

        if (node.getBody() != null) {
            long methodLength = getMethodLength(node.getBody().toString());
            if (commandLineValues.MaxCodeLength > 0) {
                if (methodLength >=commandLineValues.MinCodeLength && methodLength <= commandLineValues.MaxCodeLength) {
                    methods.add(new MethodContent(methodCode, methodName, doc));
                }
            } else {
                methods.add(new MethodContent(methodCode, methodName, doc));
            }
        }

        super.visit(node, arg);
    }

    private long getMethodLength(String code) {
        String cleanCode = code.replaceAll("\r\n", "\n").replaceAll("\t", " ");
        if (cleanCode.startsWith("{\n"))
            cleanCode = cleanCode.substring(3).trim();
        if (cleanCode.endsWith("\n}"))
            cleanCode = cleanCode.substring(0, cleanCode.length() - 2).trim();
        if (cleanCode.length() == 0) {
            return 0;
        }
        return Arrays.stream(cleanCode.split("\n"))
                .filter(line -> (line.trim() != "{" && line.trim() != "}" && line.trim() != ""))
                .filter(line -> !line.trim().startsWith("/") && !line.trim().startsWith("*")).count();
    }

    public ArrayList<MethodContent> getMethods() {
        return methods;
    }
}
