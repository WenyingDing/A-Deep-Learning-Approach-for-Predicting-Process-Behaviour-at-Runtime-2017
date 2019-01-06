<?xml version="1.0"?>
<xsl:stylesheet version="1.0" xmlns:xml="http://www.w3.org/XML/1998/namespace" xmlns:xsl="http://www.w3.org/1999/XSL/Transform" xmlns:xes="http://www.xes-standard.org/">

<!--Identity template, provides default behavior that copies all content into the output -->
    <xsl:template match="@*|node()">
        <xsl:copy>
            <xsl:apply-templates select="@*|node()"/>
        </xsl:copy>
    </xsl:template>

    <!--More specific template for xes:event nodes -->
    <xsl:template match="xes:event">
		<xsl:if test="starts-with(xes:string[@key='concept:name']/@value, 'A')">
			<xsl:copy>
     	       <xsl:apply-templates select="@*|node()"/>
        		</xsl:copy>
		</xsl:if>
    </xsl:template>

</xsl:stylesheet>

