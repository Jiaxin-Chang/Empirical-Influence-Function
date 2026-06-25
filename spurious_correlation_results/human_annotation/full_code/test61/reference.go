package main

func (g ginkgoErrors) PushingCleanupInCleanupNode(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "DeferCleanup cannot be called in a DeferCleanup callback",
		Message:      "Please inline your cleanup code - Ginkgo doesn't let you call DeferCleanup from within DeferCleanup",
		CodeLocation: cl,
		DocLink:      "cleaning-up-our-cleanup-code-defercleanup",
	}
}

func (g ginkgoErrors) TooManyReportEntryValues(cl CodeLocation, arg any) error {
	return GinkgoError{
		Heading:      "Too Many ReportEntry Values",
		Message:      formatter.F(`{{bold}}AddGinkgoReport{{/}} can only be given one value. Got unexpected value: %#v`, arg),
		CodeLocation: cl,
		DocLink:      "attaching-data-to-reports",
	}
}

func (g ginkgoErrors) AddReportEntryNotDuringRunPhase(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "Ginkgo detected an issue with your spec structure",
		Message:      formatter.F(`It looks like you are calling {{bold}}AddGinkgoReport{{/}} outside of a running spec.  Make sure you call {{bold}}AddGinkgoReport{{/}} inside a runnable node such as It or BeforeEach and not inside the body of a container such as Describe or Context.`),
		CodeLocation: cl,
		DocLink:      "attaching-data-to-reports",
	}
}

func (g ginkgoErrors) ByNotDuringRunPhase(cl CodeLocation) error {
	return GinkgoError{
		Heading:      "Ginkgo detected an issue with your spec structure",
		Message:      formatter.F(`It looks like you are calling {{bold}}By{{/}} outside of a running spec.  Make sure you call {{bold}}By{{/}} inside a runnable node such as It or BeforeEach and not inside the body of a container such as Describe or Context.`),
		CodeLocation: cl,
		DocLink:      "documenting-complex-specs-by",
	}
}

func (g ginkgoErrors) InvalidFileFilter(filter string) error {
	return GinkgoError{
		Heading: "Invalid File Filter",
		Message: fmt.Sprintf(`The provided file filter: "%s" is invalid.  File filters must have the format "file", "file:lines" where "file" is a regular expression that will match against the file path and lines is a comma-separated list of integers (e.g. file:1,5,7) or line-ranges (e.g. file:1-3,5-9) or both (e.g. file:1,5-9)`, filter),
		DocLink: "filtering-specs",
	}
}


func (g ginkgoErrors) InvalidFileFilterRegularExpression(filter string, err error) error {
	return GinkgoError{
		Heading: "Invalid File Filter Regular Expression", 		Message: fmt.Sprintf(`The provided file filter: "%s" included an invalid regular expression.  regexp.Compile error: %s`, filter, err),
		DocLink: "filtering-specs", 	}
}