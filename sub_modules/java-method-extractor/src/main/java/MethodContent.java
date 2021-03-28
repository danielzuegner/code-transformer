public class MethodContent {

    private final String code;
    private final String name;
    private final String doc;

    public MethodContent(String code, String name, String doc) {
        this.code = code;
        this.name = name;
        this.doc = doc;
    }

    public String getCode() {
        return code;
    }

    public String getName() {
        return name;
    }

    public String getDoc() {
        return doc;
    }
}
