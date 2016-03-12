package errors;
public class PathDoesNotExistException extends Exception {
    public PathDoesNotExistException(String message) {
        super(message);
    }
}
