import com.github.javaparser.Position;
import com.github.javaparser.Range;
import com.github.javaparser.ast.Node;

import java.util.ArrayList;
import java.util.List;

public class ASTNode {
    private String type;
    private Range sourceRange;
    private List<ASTNode> childNodes;


    public ASTNode(String type, Range sourceRange) {
        this.type = type;

        if (sourceRange == null) {
            this.sourceRange = null;
        } else {
            this.sourceRange = new Range(
                    new Position(sourceRange.begin.line - 1, sourceRange.begin.column),
                    new Position(sourceRange.end.line - 1, sourceRange.end.column + 1));
        }

        this.childNodes = new ArrayList<>();
    }

    public static ASTNode fromNode(Node node) {
        Range range = null;
        if (node.getRange().isPresent()) {
            range = node.getRange().get();
        }
        ASTNode parsedNode = new ASTNode(node.getClass().getSimpleName(), range);
        for (Node childNode : node.getChildNodes()) {
            parsedNode.addChildNode(ASTNode.fromNode(childNode));
        }
        return parsedNode;
    }

    public void addChildNode(ASTNode childNode) {
        this.childNodes.add(childNode);
    }
}
